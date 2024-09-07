from functools import singledispatchmethod, cached_property

import torch.nn.functional as F

from common.utils import *
from config.base_configs import HALF_RING, BIT_LEN, DEVICE, data_type, DTYPE_MAPPING, DTYPE_SCALE_MAPPING


class RingTensor(object):
    """
    自定义环上张量
    支持普通tensor的部分操作

    属性:
        tensor: 张量值
        shape: 环上张量值
        dtype: 数据规模（小数位数）
        bit_len: 转换成二进制数后，应有的二进制位长
        device: 运算所在设备
    """

    @singledispatchmethod
    def __init__(self):
        self.bit_len = None
        self.dtype = None
        self.tensor = None

    @__init__.register
    def from_tensor(self, tensor: torch.Tensor, dtype='int', device=DEVICE):
        self.tensor = tensor.to(device)
        self.dtype = dtype
        self.bit_len = BIT_LEN

    @__init__.register(int)
    @__init__.register(list)
    def from_item(self, item, dtype='int', device=DEVICE):
        self.tensor = torch.tensor(item, device=device)
        self.dtype = dtype
        self.bit_len = BIT_LEN

    @cached_property
    def scale(self):
        return DTYPE_SCALE_MAPPING[self.dtype]

    @property
    def shape(self):
        return self.tensor.shape

    @property
    def device(self):
        return self.tensor.device.type  # TODO

    @property
    def T(self):
        new_value = self.tensor.T
        return self.__class__(new_value, self.dtype, self.device)

    def __str__(self):
        return f"{self.__class__.__name__}\n value:{self.tensor} \n dtype:{self.dtype} \n scale:{self.scale}"

    def __getitem__(self, item):
        """
        获取RingTensor的某一部分
        :param item:
        :return:
        """
        return self.__class__(self.tensor[item], self.dtype).clone()

    def __setitem__(self, key, value):
        """
        设置RingTensor的某一部分
        :param key:
        :param value:
        :return:
        """
        if isinstance(value, RingTensor):
            self.tensor[key] = value.tensor.clone()
        else:
            raise TypeError(f"unsupported operand type(s) for setitem '{type(self)}' and {type(value)}")

    def __len__(self):
        """
        获取RingTensor的长度
        :return:
        """
        return len(self.tensor)

    def __invert__(self):
        """
        对RingTensor进行取反操作
        :return:
        """
        return self.__class__(~self.tensor, self.dtype, self.device)

    # add function on ring (plaintext + plaintext)
    def __add__(self, other):
        """
        对两个RingTensor进行加法操作
        :param other:
        :return:
        """
        if isinstance(other, RingTensor):
            new_value = self.tensor + other.tensor
        elif isinstance(other, int):
            new_value = self.tensor + other
        elif isinstance(other, torch.Tensor):
            new_value = self.tensor + other
        else:
            raise TypeError("unsupported operand type(s) for + 'RingTensor' and ", type(other),
                            'please convert to ring first')
        return self.__class__(new_value, self.dtype, self.device)

    # sub function on ring (plaintext - plaintext)
    def __sub__(self, other):
        """
        对两个RingTensor进行减法操作
        :param other:
        :return:
        """
        if isinstance(other, RingTensor):
            new_value = self.tensor - other.tensor
        elif isinstance(other, int):
            new_value = self.tensor - other
        elif isinstance(other, torch.Tensor):
            new_value = self.tensor - other
        else:
            raise TypeError(
                "unsupported operand type(s) for - 'RingTensor' and ", type(other), 'please convert to ring first')
        return self.__class__(new_value, self.dtype, self.device)

    # mul function on ring (plaintext * plaintext)
    def __mul__(self, other):
        """
        对两个RingTensor进行环上的乘法操作
        :param other:
        :return:
        """
        if isinstance(other, RingTensor):
            assert self.dtype == other.dtype, "dtype not equal"
            new_value = (self.tensor * other.tensor) // self.scale
        elif isinstance(other, int):
            new_value = self.tensor * other
        else:
            raise TypeError(
                "unsupported operand type(s) for * 'RingTensor' and ", type(other), 'please convert to ring first')
        return self.__class__(new_value, self.dtype, self.device)

    # mod function on ring (plaintext % int)
    def __mod__(self, other):
        """
        对两个RingTensor进行取模操作
        :param other:
        :return:
        """
        if isinstance(other, int):
            new_value = (self.tensor % other)
        else:
            raise TypeError(
                "unsupported operand type(s) for % 'RingTensor' and ", type(other), 'please convert to ring first')
        return self.__class__(new_value, self.dtype, self.device)

    def __matmul__(self, other):
        """
        对两个RingTensor矩阵相乘
        :param other:
        :return:
        """
        if isinstance(other, RingTensor):
            assert self.dtype == other.dtype, "dtype not equal"
            if self.device == 'cuda':
                new_value = cuda_matmul(self.tensor, other.tensor) // self.scale
            else:
                new_value = torch.matmul(self.tensor, other.tensor) // self.scale
        else:
            raise TypeError(
                "unsupported operand type(s) for @ 'RingTensor' and ", type(other), 'please convert to ring first'
            )
        return self.__class__(new_value, self.dtype, self.device)

    def __truediv__(self, other):
        if isinstance(other, RingTensor):
            assert self.scale == other.scale, "data must have the same scale"
            new_value = self.tensor / other.tensor * self.scale
        elif isinstance(other, torch.Tensor):
            new_value = self.tensor / other
        elif isinstance(other, int):
            new_value = self.tensor / other
        else:
            raise TypeError("unsupported operand type(s) for / 'RingTensor' and ", type(other))
        return self.__class__(torch.round(new_value), self.dtype, self.device)

    def __floordiv__(self, other):
        if isinstance(other, RingTensor):
            assert self.scale == other.scale, "data must have the same scale"
            new_value = self.tensor // other.tensor * self.scale
        elif isinstance(other, torch.Tensor):
            new_value = self.tensor // other
        elif isinstance(other, int):
            new_value = self.tensor // other
        else:
            raise TypeError("unsupported operand type(s) for // 'RingTensor' and ", type(other))
        return self.__class__(new_value, self.dtype, self.device)

    # neg function on ring (-plaintext)
    def __neg__(self):
        """
        对RingTensor进行取负操作
        :return:
        """
        new_value = -self.tensor
        return self.__class__(new_value, self.dtype, self.device)

    # TODO: Unable to use comparison of real number fields with ring elements
    # eq function on ring (plaintext == plaintext)
    def __eq__(self, other):
        """
        判断两个RingTensor是否相等
        :param other:
        :return:
        """
        if isinstance(other, RingTensor):
            new_value = (self.tensor == other.tensor)
        else:
            raise TypeError(
                "unsupported operand type(s) for == 'RingTensor' and ", type(other), 'please convert to ring first')
        return new_value

    # ne function on ring (plaintext != plaintext)
    def __ne__(self, other):
        """
        判断两个RingTensor是否不相等
        :param other:
        :return:
        """
        if isinstance(other, RingTensor):
            return self.tensor != other.tensor
        else:
            raise TypeError(
                "unsupported operand type(s) for != 'RingTensor' and ", type(other), 'please convert to ring first')

    # gt function on ring (plaintext > plaintext)
    def __gt__(self, other):
        """
        判断一个RingTensor是否大于另一个RingTensor
        :param other:
        :return:
        """
        if isinstance(other, RingTensor):
            new_value = (self.tensor > other.tensor)
        else:
            raise TypeError(
                "unsupported operand type(s) for > 'RingTensor' and ", type(other), 'please convert to ring first')
        return new_value

    # ge function on ring (plaintext >= plaintext)
    def __ge__(self, other):
        """
        判断一个RingTensor是否大于或等于另一个RingTensor
        :param other:
        :return:
        """
        if isinstance(other, RingTensor):
            new_value = (self.tensor >= other.tensor)
        else:
            raise TypeError(
                "unsupported operand type(s) for >= 'RingTensor' and ", type(other), 'please convert to ring first')
        return new_value

    # lt function on ring (plaintext < plaintext)
    def __lt__(self, other):
        """
        判断一个RingTensor是否小于另一个RingTensor
        :param other:
        :return:
        """
        if isinstance(other, RingTensor):
            new_value = (self.tensor < other.tensor)
        else:
            raise TypeError(
                "unsupported operand type(s) for < 'RingTensor' and ", type(other), 'please convert to ring first')
        return new_value

    # le function on ring (plaintext <= plaintext)
    def __le__(self, other):
        """
        判断一个RingTensor是否小于或等于另一个RingTensor
        :param other:
        :return:
        """
        if isinstance(other, RingTensor):
            new_value = (self.tensor <= other.tensor)
        else:
            raise TypeError(
                "unsupported operand type(s) for <= 'RingTensor' and ", type(other), 'please convert to '
                                                                                     'ring first')
        return new_value

    def __xor__(self, other):
        """
        对两个RingTensor进行异或操作
        :param other:
        :return:
        """
        if isinstance(other, RingTensor):
            new_value = (self.tensor ^ other.tensor)
        elif isinstance(other, int):
            # convert int to torch.tensor
            tensor = torch.tensor([other], dtype=data_type, device=self.device)
            new_value = (self.tensor ^ tensor)
        elif isinstance(other, torch.Tensor):
            new_value = (self.tensor ^ other)
        else:
            raise TypeError(
                "unsupported operand type(s) for ^ 'RingTensor' and ", type(other), 'please convert to ring first')
        return self.__class__(new_value, self.dtype)

    def __or__(self, other):
        if isinstance(other, int):
            return self.__class__(self.tensor | other, self.dtype)
        else:
            raise TypeError(
                "unsupported operand type(s) for << 'RingTensor' and ", type(other), 'please convert to ring first')

    def __and__(self, other):
        if isinstance(other, int):
            return self.__class__(self.tensor & other, self.dtype)
        else:
            raise TypeError(
                "unsupported operand type(s) for << 'RingTensor' and ", type(other), 'please convert to ring first')

    def __rshift__(self, other):
        """
        对RingTensor进行右移操作
        :param other:
        :return:
        """
        if isinstance(other, int):
            return self.__class__(self.tensor >> other, self.dtype)
        if isinstance(other, RingTensor):
            return self.__class__(self.tensor >> other.tensor, self.dtype)
        else:
            raise TypeError(
                "unsupported operand type(s) for >> 'RingTensor' and ", type(other), 'please convert to ring first')

    def __lshift__(self, other):
        """
        对RingTensor进行左移操作
        :param other:
        :return:
        """
        if isinstance(other, int):
            return self.__class__(self.tensor << other, self.dtype)
        else:
            raise TypeError(
                "unsupported operand type(s) for << 'RingTensor' and ", type(other), 'please convert to ring first')

    @classmethod
    def convert_to_ring(cls, item):  # todo device
        """
        采用静态方法：将torch.Tensor对象转换为RingTensor对象
        :param item:
        :return:
        """
        if isinstance(item, (int, list)):
            item = torch.tensor(item)
        assert isinstance(item, torch.Tensor), f"unsupported data type(s): {type(item)}"
        scale = DTYPE_MAPPING[item.dtype]
        v = torch.round(item * scale) if scale != 1 else item
        dtype = 'int' if scale == 1 else 'float'
        r = cls(v.to(data_type), dtype=dtype, device=item.device)
        return r

    @classmethod
    def random(cls, shape, dtype='int', device=DEVICE, down_bound=-HALF_RING, upper_bound=HALF_RING - 1):
        """
        生成一个随机RingTensor
        :param device:
        :param shape:
        :param down_bound:随机数下界，可取
        :param upper_bound:随机数上界，不可取
        :param dtype:
        :return:
        """
        v = torch.randint(down_bound, upper_bound, shape, dtype=data_type, device=device)
        return cls(v, dtype, device)

    @classmethod
    def load_from_file(cls, file_path):
        """
        从文件中加载RingTensor
        :param file_path:
        :return:
        """
        return cls(torch.load(file_path))

    @classmethod
    def load_from_value(cls, v, dtype='int'):
        """
        从给定的值中加载RingTensor
        :param v:
        :param dtype:
        :return:
        """
        return cls(v, dtype)

    @classmethod
    def mul(cls, x, y):
        return cls(x.tensor * y.tensor, x.dtype)

    @classmethod
    def matmul(cls, x, y):
        if x.device != y.device:
            raise TypeError(
                "Expected all ring tensors to be on the same device, but found at least two devices,"
                + f" {x.device} and {y.device}!")
        if x.device == 'cpu':
            return cls(torch.matmul(x.tensor, y.tensor), x.dtype)
        if x.device in ('cuda', 'cuda:0', 'cuda:1'):
            return cls(cuda_matmul(x.tensor, y.tensor), x.dtype)

    @classmethod
    def exp(cls, x):
        return cls.convert_to_ring(torch.exp(x.tensor // x.scale) * x.scale)

    @classmethod
    def empty(cls, size, dtype='int', device=DEVICE):
        """
        获取一个空的RingTensor
        :return:
        """
        return cls(torch.empty(size, dtype=data_type), dtype, device)

    @classmethod
    def empty_like(cls, tensor):
        """
        获取一个和给定的tensor相同形状的空的RingTensor
        :return:
        """
        if isinstance(tensor, RingTensor):
            return cls(torch.empty_like(tensor.tensor), tensor.dtype, tensor.device)
        else:
            raise TypeError("unsupported operand type(s) for empty_like 'RingTensor' and ", type(tensor))

    @classmethod
    def zeros(cls, size, dtype='int', device=DEVICE):
        """
        获取一个全零的RingTensor
        :return:
        """
        return cls(torch.zeros(size, dtype=data_type, device=device), dtype)

    @classmethod
    def zeros_like(cls, tensor, dtype='int', device=DEVICE):
        """
        获取一个和给定的tensor相同形状的全零的RingTensor
        :return:
        """
        if isinstance(tensor, RingTensor):
            return cls(torch.zeros_like(tensor.tensor), tensor.dtype, device)
        elif isinstance(tensor, torch.Tensor):
            return cls(torch.zeros_like(tensor), dtype, device)
        else:
            raise TypeError("unsupported operand type(s) for zeros_like 'RingTensor' and ", type(tensor))

    @classmethod
    def ones(cls, size, dtype='int', device=DEVICE):
        """
        获取一个全1的RingTensor
        :return:
        """
        scale = DTYPE_SCALE_MAPPING[dtype]
        return cls(torch.ones(size, dtype=data_type, device=device) * scale, dtype)

    @classmethod
    def ones_like(cls, tensor, dtype='int', device=DEVICE):
        """
        获取一个和给定的tensor相同形状的全1的RingTensor
        :return:
        """

        if isinstance(tensor, RingTensor):
            return cls(torch.ones_like(tensor.tensor) * tensor.scale, tensor.dtype, device)
        elif isinstance(tensor, torch.Tensor):
            scale = DTYPE_SCALE_MAPPING[dtype]
            return cls(torch.ones_like(tensor) * scale, dtype, device)
        else:
            raise TypeError("unsupported operand type(s) for ones_like 'RingTensor' and ", type(tensor))

    @classmethod
    def full(cls, size, fill_value, device=DEVICE):
        """
        获取一个全fill_value的RingTensor
        :return:
        """
        return cls.convert_to_ring(torch.full(size, fill_value, device=device))

    @classmethod
    def full_like(cls, tensor, fill_value, device=DEVICE):
        """
        获取一个和给定的tensor相同形状的全fill_value的RingTensor
        :return:
        """
        if isinstance(tensor, RingTensor):
            return cls.full(tensor.shape, fill_value, device=tensor.device)
        elif isinstance(tensor, torch.Tensor):
            return cls.convert_to_ring(torch.full_like(tensor, fill_value, device=tensor.device))
        else:
            raise TypeError("unsupported operand type(s) for full_like 'RingTensor' and ", type(tensor))

    @classmethod
    def cat(cls, tensors, dim=0):
        """
        在RingTensor上使用cat方法拼接新的张量
        :param tensors: 别的张量
        :param dim: 维度
        :return:
        """
        # TODO: 如果dtype不同，是否支持
        if isinstance(tensors[0], RingTensor):
            return cls(fast_concat([t.tensor for t in tensors], dim), tensors[0].dtype)
        else:
            raise TypeError(f"unsupported operand type(s) for cat '{cls.__name__}' and {type(tensors[0])}")

    @classmethod
    def diagonal(cls, input, offset=0, dim1=0, dim2=1):
        return cls(torch.diagonal(input.tensor, offset, dim1, dim2), input.dtype, input.device)

    @classmethod
    def roll(cls, input, shifts, dims=0):
        return cls(torch.roll(input.tensor, shifts=shifts, dims=dims), input.dtype, input.device)

    @classmethod
    def arange(cls, start, end, step=1, dtype='int', device=DEVICE):
        """
        获取一个从start到end的RingTensor
        :return:
        """
        return cls(torch.arange(start, end, step, dtype=data_type, device=device), dtype)

    @classmethod
    def row_shift(cls, input, shifts):
        return cls(rows_shift(input.tensor, offsets=shifts), input.dtype, input.device)

    # convert ring field to real field
    def convert_to_real_field(self):
        """
        将RingTensor对象从环域转换到实数域
        :return:
        """
        return self.tensor / self.scale

    def sum(self, dim=0):
        """
        沿axis轴对RingTensor求和
        :param dim:
        :return:
        """
        new_value = torch.sum(self.tensor, dim=dim)
        return self.__class__(new_value, self.dtype)

    def to(self, device):
        """
        将RingTensor转移到指定设备上
        :param device:
        :return:
        """
        self.tensor = self.tensor.to(device)
        return self

    def save(self, file_path):
        """
        保存RingTensor到文件
        :param file_path:
        :return:
        """
        torch.save(self.tensor, file_path)
        print("Successfully save to ", file_path)

    # clone a ring tensor to new ring tensor
    def clone(self):
        new = self.__class__(self.tensor.clone(), self.dtype, self.device)
        new.bit_len = self.bit_len
        return new

    # get bit
    def get_bit(self, item):
        """
        获取RingTensor的某一bit
        :param item:
        :return:
        """
        assert (self.bit_len >> item >= 0), "bit index out of range"
        return (self.tensor >> item) & 1
        # return RingTensor((self.tensor >> item) & 1)

    # new tensor maybe ?
    def reshape(self, *shape):
        """
        重塑RingTensor的形状
        :param shape:
        :return:
        """
        new = self.__class__(self.tensor.reshape(*shape), self.dtype, self.device)
        new.bit_len = self.bit_len
        return new

    def img2col(self, k_size: int, stride: int):
        """
        Img2Col卷积池化加速算法，对图像张量变形以适配卷积，池化等操作

        :param k_size: 卷积核，池化核大小
        :param stride: 卷积，池化步长
        :return: 展开后的特征张量
        """

        img = self.tensor

        batch, channel, height, width = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
        out_h = (height - k_size) // stride + 1
        out_w = (width - k_size) // stride + 1
        kw = kh = k_size
        out_size = out_w * out_h
        col = torch.zeros(size=(batch, channel, kw * kh, out_size), dtype=data_type, device=self.device)
        for y in range(out_h):
            y_start = y * stride
            y_end = y_start + kh
            for x in range(out_w):
                x_start = x * stride
                x_end = x_start + kw
                col[..., 0:, y * out_w + x] = img[..., y_start:y_end, x_start:x_end].reshape(batch, channel, kh * kw)

        col = self.__class__(col, self.dtype, self.device)
        return col, batch, out_size, channel

    def repeat_interleave(self, repeats, dim):
        return self.__class__(self.tensor.repeat_interleave(repeats, dim), self.dtype)

    def repeat(self, *sizes):
        return self.__class__(self.tensor.repeat(*sizes), self.dtype)

    def transpose(self, dim0, dim1):
        return self.__class__(self.tensor.transpose(dim0, dim1), self.dtype)

    def pad(self, pad, mode='constant', value=0):
        return self.__class__(F.pad(self.tensor, pad, mode, value), self.dtype)

    def squeeze(self, dim):
        return self.__class__(self.tensor.squeeze(dim), self.dtype)

    def unsqueeze(self, dim):
        return self.__class__(self.tensor.unsqueeze(dim), self.dtype)

    def size(self, dim=None):
        if dim is None:
            return self.shape
        else:
            return self.shape[dim]

    def view(self, *args):
        view = self.__class__(self.tensor.view(*args), self.dtype)
        view.bit_len = self.bit_len
        return view

    def flatten(self, dim=0):
        new = self.__class__(self.tensor.flatten(dim), self.dtype)
        new.bit_len = self.bit_len
        return new

    def permute(self, dims):
        new = self.__class__(self.tensor.permute(dims=dims), self.dtype)
        new.bit_len = self.bit_len
        return new

    def tolist(self):
        return self.tensor.tolist()

    def numel(self):
        return self.tensor.numel()

    def signbit(self):
        # 在本系统中，使用signed int，所以最高位不能用位移来获取，直接与0进行大小比较
        msb = torch.signbit(self.tensor) + 0
        return self.__class__(msb, self.dtype)

    def bit_slice(self, start, end):
        # 获取RingTensor中每一个元素的bit分割
        assert (self.bit_len >> start >= 0), "bit index out of range"
        assert (self.bit_len >> end >= 0), "bit index out of range"

        if start == 0 and end == self.bit_len:
            return self

        if end == self.bit_len:
            return self >> start

        shift_right = self >> start
        new_end = end - start
        mask = (1 << (new_end + 1)) - 1
        masked_value = shift_right & mask
        return masked_value

        # mask = ((1 << (end - start + 1)) - 1) << start
        # result = (self.tensor & mask) >> start
        # new = RingTensor(result)
        # new.bit_len = self.bit_len
        # return new
