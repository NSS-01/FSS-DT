"""
Replicated Secret Sharing
"""
from common.utils.tensor_utils import list_rotate
from crypto.primitives.function_secret_sharing.dicf import DICFKey
from crypto.protocols.comparison.cmp_sigma import CMPSigma, CMPSigmaKey
from crypto.primitives.oblivious_transfer.oblivious_transfer_aby3 import OT
from crypto.tensor.RingTensor import RingTensor
from config.base_configs import *


# For 3 parties only
class ReplicatedSecretSharing(object):
    """
    支持3-2算术秘密共享的自定义tensor类
    属性:
        replicated_shared_tensor: 其张量值，每一方拥有两个分享值
        party: 所属参与方
    """

    def __init__(self, replicated_shared_tensor, party):
        self.replicated_shared_tensor = replicated_shared_tensor  # should be a list which have two elements
        self.party = party
        self.device = None
        self.shape = self.replicated_shared_tensor[0].shape

    def __str__(self):
        return "[{}\n value 1:{},\n value 2:{},\nparty:{}]".format(self.__class__.__name__,
                                                                   self.replicated_shared_tensor[0].tensor,
                                                                   self.replicated_shared_tensor[1].tensor,
                                                                   self.party.party_id)

    # sum function: sum the tensor 1 and tensor 2 locally
    def sum(self, axis=0):
        """
        沿着指定的轴对存储在 replicated_shared_tensor 中的两个张量分别进行求和。

        :param axis: 要进行求和操作的轴。
        :return: 一个新的 ReplicatedSecretSharing 实例，其中包含两个求和后的张量和当前的 party 信息。
        """
        new_tensor1 = self.replicated_shared_tensor[0].sum(axis)
        new_tensor2 = self.replicated_shared_tensor[1].sum(axis)
        return ReplicatedSecretSharing([new_tensor1, new_tensor2], self.party)

    def reshape(self, *shape):
        """
        重塑replicated_shared_tensor中两个张量的形状。

        :param shape:重塑的形状元组。
        :return:重塑后的replicated_shared_tensor。
        """
        rss0 = self.replicated_shared_tensor[0].reshape(*shape)
        rss1 = self.replicated_shared_tensor[1].reshape(*shape)
        return ReplicatedSecretSharing([rss0, rss1], self.party)

    def size(self, dim=None):
        if dim is None:
            return self.shape
        else:
            return self.shape[dim]

    @staticmethod
    def share(tensor: RingTensor):
        """
        对输入RingTensor进行三方复制秘密分享。

        :param tensor: 进行秘密分享的输入数据张量,类型为RingTensor。
        :return: 复制秘密分享后的分享份额列表，包含三个RingTensor的二元列表。
        """
        shares = []
        x_0 = RingTensor.random(tensor.shape, tensor.dtype, tensor.scale)
        x_1 = RingTensor.random(tensor.shape, tensor.dtype, tensor.scale)
        x_2 = tensor - x_0 - x_1
        shares.append([x_0, x_1])
        shares.append([x_1, x_2])
        shares.append([x_2, x_0])
        return shares

    @staticmethod
    def load_from_ring_tensor(replicated_shared_tensor, party):
        """
        从给定的RingTensor列表和party中加载ReplicatedSecretSharing。

        :param replicated_shared_tensor: 由两个RingTensor实例组成的列表。
        :param party: 一个参与方的party实例。
        :return: 加载后的ReplicatedSecretSharing实例。
        """
        return ReplicatedSecretSharing(replicated_shared_tensor, party)

    def restore(self):
        """
        基于三方复制秘密分享的数据张量的明文值恢复。

        :return: 恢复后的数据张量，类型为RingTensor。
        """
        # 发送部分
        self.party.send_ring_tensor_to((self.party.party_id + 1) % 3, self.replicated_shared_tensor[0])
        # 接收部分
        other = self.party.receive_ring_tensor_from((self.party.party_id + 2) % 3)
        return self.replicated_shared_tensor[0] + self.replicated_shared_tensor[1] + other

    # def reshare33(self, value):
    #     self.party.send_ring_tensor_to((self.party.party_id + 2) % 3, value)
    #     other = self.party.receive_ring_tensor_from((self.party.party_id + 1) % 3)
    #     return ReplicatedSecretSharing([value, other], self.party)

    @staticmethod
    def reshare33(value, party):
        """
        基于三个参与方的三三复制秘密分享。

        :param value: 当前参与方所持有的秘密分享的数据张量，类型为RingTensor。
        :param party: 当前参与方的party实例。
        :param party: 当前参与方的party实例。
        :return: 经三三复制秘密分享后，当前参与方所持有的部分，类型为ReplicatedSecretSharing。
        """

        c = ReplicatedSecretSharing.rand_like(value, party)
        value = value + c.replicated_shared_tensor[0] - c.replicated_shared_tensor[1]
        party.send_ring_tensor_to((party.party_id + 2) % 3, value)
        other = party.receive_ring_tensor_from((party.party_id + 1) % 3)
        return ReplicatedSecretSharing([value, other], party)

    # add function: each party add its share of the tensor locally
    def __add__(self, other):
        new_tensor1 = None
        new_tensor2 = None
        if isinstance(other, ReplicatedSecretSharing):
            new_tensor1 = self.replicated_shared_tensor[0] + other.replicated_shared_tensor[0]
            new_tensor2 = self.replicated_shared_tensor[1] + other.replicated_shared_tensor[1]
        elif isinstance(other, RingTensor):
            if self.party.party_id == 0:
                new_tensor1 = self.replicated_shared_tensor[0] + other
                new_tensor2 = self.replicated_shared_tensor[1]
            elif self.party.party_id == 1:
                new_tensor1 = self.replicated_shared_tensor[0]
                new_tensor2 = self.replicated_shared_tensor[1]
            else:
                new_tensor1 = self.replicated_shared_tensor[0]
                new_tensor2 = self.replicated_shared_tensor[1] + other
        else:
            TypeError("unsupported operand type(s) for + 'ReplicatedSecretSharing' and ", type(other))
        return ReplicatedSecretSharing([new_tensor1, new_tensor2], self.party)

        # add function: each party add its share of the tensor locally

    def __sub__(self, other):
        new_tensor1 = None
        new_tensor2 = None
        if isinstance(other, ReplicatedSecretSharing):
            new_tensor1 = self.replicated_shared_tensor[0] - other.replicated_shared_tensor[0]
            new_tensor2 = self.replicated_shared_tensor[1] - other.replicated_shared_tensor[1]
        elif isinstance(other, RingTensor):
            if self.party.party_id == 0:
                new_tensor1 = self.replicated_shared_tensor[0] - other
                new_tensor2 = self.replicated_shared_tensor[1]
            elif self.party.party_id == 1:
                new_tensor1 = self.replicated_shared_tensor[0]
                new_tensor2 = self.replicated_shared_tensor[1]
            else:
                new_tensor1 = self.replicated_shared_tensor[0]
                new_tensor2 = self.replicated_shared_tensor[1] - other
        else:
            TypeError("unsupported operand type(s) for - 'ReplicatedSecretSharing' and ", type(other))
        return ReplicatedSecretSharing([new_tensor1, new_tensor2], self.party)

    def __mul__(self, other):
        if isinstance(other, ReplicatedSecretSharing):
            z_shared = self.replicated_shared_tensor[0].tensor * other.replicated_shared_tensor[0].tensor + \
                       self.replicated_shared_tensor[0].tensor * other.replicated_shared_tensor[1].tensor + \
                       self.replicated_shared_tensor[1].tensor * other.replicated_shared_tensor[0].tensor
            z_shared = RingTensor(z_shared, self.replicated_shared_tensor[0].dtype,
                                  self.replicated_shared_tensor[0].scale, self.device)

            result = ReplicatedSecretSharing.reshare33(z_shared, self.party)

            if self.replicated_shared_tensor[0].dtype == "float":
                scale = self.replicated_shared_tensor[0].scale
                result = truncate(result, scale)

            return result
            # return z_shared
        elif isinstance(other, RingTensor) or isinstance(other, int):
            new_tensor1 = self.replicated_shared_tensor[0] * other
            new_tensor2 = self.replicated_shared_tensor[1] * other
            return ReplicatedSecretSharing([new_tensor1, new_tensor2], self.party)
        else:
            TypeError("unsupported operand type(s) for * 'ReplicatedSecretSharing' and ", type(other))

    def __matmul__(self, other):
        if isinstance(other, ReplicatedSecretSharing):
            z_shared = RingTensor.matmul(self.replicated_shared_tensor[0], other.replicated_shared_tensor[0]) + \
                       RingTensor.matmul(self.replicated_shared_tensor[0], other.replicated_shared_tensor[1]) + \
                       RingTensor.matmul(self.replicated_shared_tensor[1], other.replicated_shared_tensor[0])

            result = ReplicatedSecretSharing.reshare33(z_shared, party=self.party)

            if self.replicated_shared_tensor[0].dtype == "float":
                scale = self.replicated_shared_tensor[0].scale
                result = truncate(result, scale)

            return result
        else:
            TypeError("unsupported operand type(s) for @ 'ReplicatedSecretSharing' and ", type(other))

    def __ge__(self, other):
        if isinstance(other, ReplicatedSecretSharing):
            return secure_ge(self, other)
        else:
            raise TypeError("unsupported operand type(s) for >= 'ReplicatedSecretSharing' and ", type(other))

    def __le__(self, other):
        if isinstance(other, ReplicatedSecretSharing):
            return secure_ge(other, self)
        else:
            raise TypeError("unsupported operand type(s) for <= 'ReplicatedSecretSharing' and ", type(other))

    def __gt__(self, other):
        if isinstance(other, ReplicatedSecretSharing):
            ge = secure_ge(other, self)
            return (ge - RingTensor.ones_like(ge.replicated_shared_tensor[0])) * -1
        else:
            raise TypeError("unsupported operand type(s) for > 'ReplicatedSecretSharing' and ", type(other))

    def __lt__(self, other):
        if isinstance(other, ReplicatedSecretSharing):
            ge = secure_ge(self, other)
            return (ge - RingTensor.ones_like(ge.replicated_shared_tensor[0])) * -1
        else:
            raise TypeError("unsupported operand type(s) for > 'ReplicatedSecretSharing' and ", type(other))

    def __xor__(self, other):
        if isinstance(other, ReplicatedSecretSharing):
            new_tensor1 = self.replicated_shared_tensor[0] ^ other.replicated_shared_tensor[0]
            new_tensor2 = self.replicated_shared_tensor[1] ^ other.replicated_shared_tensor[1]
            return ReplicatedSecretSharing([new_tensor1, new_tensor2], self.party)
        else:
            raise TypeError("unsupported operand type(s) for ^ 'ReplicatedSecretSharing' and ", type(other))

    def __rshift__(self, other):
        if isinstance(other, int):
            new_tensor1 = self.replicated_shared_tensor[0] >> other
            new_tensor2 = self.replicated_shared_tensor[1] >> other
            return ReplicatedSecretSharing([new_tensor1, new_tensor2], self.party)
        else:
            raise TypeError("unsupported operand type(s) for >> 'ReplicatedSecretSharing' and ", type(other))

    def __lshift__(self, other):
        if isinstance(other, int):
            new_tensor1 = self.replicated_shared_tensor[0] << other
            new_tensor2 = self.replicated_shared_tensor[1] << other
            return ReplicatedSecretSharing([new_tensor1, new_tensor2], self.party)
        else:
            raise TypeError("unsupported operand type(s) for << 'ReplicatedSecretSharing' and ", type(other))

    def __getitem__(self, item):
        new_replicated_tensor = [self.replicated_shared_tensor[0][item], self.replicated_shared_tensor[1][item]]
        return ReplicatedSecretSharing(new_replicated_tensor, self.party).clone()

    def __setitem__(self, key, value):
        self.replicated_shared_tensor[0][key] = value.replicated_shared_tensor[0].clone()
        self.replicated_shared_tensor[1][key] = value.replicated_shared_tensor[1].clone()

        # print("self restore",self.restore())

    @staticmethod
    def rand(num_of_value, party):
        # for i
        r_0 = party.prg_0.random(num_of_value)
        # for i+1
        r_1 = party.prg_1.random(num_of_value)
        # 由于我们直接从环上生成的随机数，因此不需要convert
        r_0_ring = RingTensor(r_0, party.dtype, party.scale)
        r_1_ring = RingTensor(r_1, party.dtype, party.scale)
        r = ReplicatedSecretSharing([r_0_ring, r_1_ring], party)
        return r

    @staticmethod
    def rand_like(x, party):
        r = ReplicatedSecretSharing.rand(x.numel(), party)
        r = r.reshape(x.shape)
        if isinstance(x, ReplicatedSecretSharing):
            r.replicated_shared_tensor[0].dtype = x.replicated_shared_tensor[0].dtype
            r.replicated_shared_tensor[1].dtype = x.replicated_shared_tensor[1].dtype
            r.replicated_shared_tensor[0].scale = x.replicated_shared_tensor[0].scale
            r.replicated_shared_tensor[1].scale = x.replicated_shared_tensor[1].scale
        if isinstance(x, RingTensor):
            r.replicated_shared_tensor[0].dtype = x.dtype
            r.replicated_shared_tensor[1].dtype = x.dtype
            r.replicated_shared_tensor[0].scale = x.scale
            r.replicated_shared_tensor[1].scale = x.scale
        return r

    @staticmethod
    def empty(shape, dtype, scale, party):
        r0 = RingTensor.empty(shape, dtype, scale)
        r1 = RingTensor.empty(shape, dtype, scale)
        return ReplicatedSecretSharing([r0, r1], party)

    @staticmethod
    def empty_like(x, party):
        if isinstance(x, ReplicatedSecretSharing):
            r0 = RingTensor.empty_like(x.replicated_shared_tensor[0])
            r1 = RingTensor.empty_like(x.replicated_shared_tensor[1])
            return ReplicatedSecretSharing([r0, r1], party)
        elif isinstance(x, RingTensor):
            r0 = RingTensor.empty_like(x)
            r1 = RingTensor.empty_like(x)
            return ReplicatedSecretSharing([r0, r1], party)

    @staticmethod
    def zeros(shape, dtype, scale, party):
        r0 = RingTensor.zeros(shape, dtype, scale)
        r1 = RingTensor.zeros(shape, dtype, scale)
        return ReplicatedSecretSharing([r0, r1], party)

    @staticmethod
    def zeros_like(x, party):
        if isinstance(x, ReplicatedSecretSharing):
            r0 = RingTensor.zeros_like(x.replicated_shared_tensor[0])
            r1 = RingTensor.zeros_like(x.replicated_shared_tensor[1])
            return ReplicatedSecretSharing([r0, r1], party)
        elif isinstance(x, RingTensor):
            r0 = RingTensor.zeros_like(x)
            r1 = RingTensor.zeros_like(x)
            return ReplicatedSecretSharing([r0, r1], party)

    def save(self, file_path):
        """
        将replicated_share_tensor保存到文件中

        :param file_path: 文件路径
        """
        self.replicated_shared_tensor[0].save(file_path + '_0' + '.pth')
        self.replicated_shared_tensor[1].save(file_path + '_1' + '.pth')

    @staticmethod
    def load_from_file(file_path, party):
        r0 = RingTensor.load_from_file(file_path + '_0' + '.pth')
        r1 = RingTensor.load_from_file(file_path + '_1' + '.pth')
        return ReplicatedSecretSharing([r0, r1], party)

    def to(self, device):
        """
        将RingTensor转移到指定设备上
        :param device:
        :return:
        """
        self.replicated_shared_tensor[0] = self.replicated_shared_tensor[0].to(device)
        self.replicated_shared_tensor[1] = self.replicated_shared_tensor[1].to(device)
        self.device = device
        return self

    def pad(self, pad, mode='constant', value=0):
        new_rss1 = self.replicated_shared_tensor[0].pad(pad, mode, value)
        new_rss2 = self.replicated_shared_tensor[1].pad(pad, mode, value)

        return ReplicatedSecretSharing([new_rss1, new_rss2], self.party)

    def repeat_interleave(self, repeat_times, dim):
        new_rss1 = self.replicated_shared_tensor[0].repeat_interleave(repeat_times, dim=dim)
        new_rss2 = self.replicated_shared_tensor[1].repeat_interleave(repeat_times, dim=dim)
        return ReplicatedSecretSharing([new_rss1, new_rss2], self.party)

    def squeeze(self, dim):
        new_rss1 = self.replicated_shared_tensor[0].squeeze(dim=dim)
        new_rss2 = self.replicated_shared_tensor[1].squeeze(dim=dim)
        return ReplicatedSecretSharing([new_rss1, new_rss2], self.party)

    def unsqueeze(self, dim):
        new_rss1 = self.replicated_shared_tensor[0].unsqueeze(dim=dim)
        new_rss2 = self.replicated_shared_tensor[1].unsqueeze(dim=dim)
        return ReplicatedSecretSharing([new_rss1, new_rss2], self.party)

    def clone(self):
        return ReplicatedSecretSharing(
            [self.replicated_shared_tensor[0].clone(), self.replicated_shared_tensor[1].clone()], self.party)

    def view(self, *args):
        r0_view = self.replicated_shared_tensor[0].view(*args)
        r1_view = self.replicated_shared_tensor[1].view(*args)
        return ReplicatedSecretSharing([r0_view, r1_view], self.party)

    @staticmethod
    def gen_and_share(r_tensor, party):
        r0, r1, r2 = ReplicatedSecretSharing.share(r_tensor)
        r = ReplicatedSecretSharing(r0, party)
        r1 = ReplicatedSecretSharing(r1, party)
        r2 = ReplicatedSecretSharing(r2, party)

        party.send_rss_to((party.party_id + 1) % 3, r1)
        party.send_rss_to((party.party_id + 2) % 3, r2)

        return r

    def flatten(self):
        rss_0 = self.replicated_shared_tensor[0].flatten()
        rss_1 = self.replicated_shared_tensor[1].flatten()
        return ReplicatedSecretSharing([rss_0, rss_1], self.party)

    def numel(self):
        return self.replicated_shared_tensor[0].numel()

    def cat(self, other, dim=0):
        if isinstance(other, ReplicatedSecretSharing):
            rss_0 = self.replicated_shared_tensor[0].cat(other.replicated_shared_tensor[0], dim=dim)
            rss_1 = self.replicated_shared_tensor[1].cat(other.replicated_shared_tensor[1], dim=dim)

            return ReplicatedSecretSharing([rss_0, rss_1], self.party)
        else:
            raise TypeError("unsupported operand type(s) for cat 'ReplicatedSecretSharing' and ", type(other))

    def transpose(self, dim0, dim1):
        rss_0 = self.replicated_shared_tensor[0].transpose(dim0, dim1)
        rss_1 = self.replicated_shared_tensor[1].transpose(dim0, dim1)

        return ReplicatedSecretSharing([rss_0, rss_1], self.party)

    def T(self):
        rss_0 = self.replicated_shared_tensor[0].T()
        rss_1 = self.replicated_shared_tensor[1].T()

        return ReplicatedSecretSharing([rss_0, rss_1], self.party)

    @staticmethod
    def diagonal(input, offset=0, dim1=0, dim2=1):
        r0 = RingTensor.diagonal(input.replicated_shared_tensor[0], offset, dim1, dim2)
        r1 = RingTensor.diagonal(input.replicated_shared_tensor[1], offset, dim1, dim2)

        return ReplicatedSecretSharing([r0, r1], input.party)


def img2col_for_conv(rss_img: ReplicatedSecretSharing, k_size: int, stride: int):
    """
    适配卷积层的张量变形方法

    :param rss_img:
    :param k_size:
    :param stride:
    :return:
    """

    img = rss_img.replicated_shared_tensor[0].cat(rss_img.replicated_shared_tensor[1])

    col, batch, out_size, _ = img.img2col(k_size, stride)

    return ReplicatedSecretSharing(
        [col[0].reshape((batch // 2, -1, out_size)), col[1].reshape((batch // 2, -1, out_size))], rss_img.party)


def img2col_for_pool(rss_img: ReplicatedSecretSharing, k_size: int, stride: int):
    """
    适用于池化层的张量变形方法
    :param rss_img:
    :param k_size:
    :param stride:
    :return:
    """

    img = rss_img.replicated_shared_tensor[0].cat(rss_img.replicated_shared_tensor[1])

    col, batch, out_size, channel = img.img2col(k_size, stride)

    return ReplicatedSecretSharing(
        [col[0].reshape((batch // 2, channel, -1, out_size)), col[1].reshape((batch // 2, channel, -1, out_size))],
        rss_img.party)


def secure_ge(x: ReplicatedSecretSharing, y: ReplicatedSecretSharing):
    z = x - y
    party = x.party

    # 令P0方生成密钥，P1和P2方计算大小比较结果
    # TODO: 这里密钥分发一轮通信要在离线阶段完成
    if party.party_id == 0:
        k0, k1 = party.compare_key_provider.get_keys_by_pointer(z.numel(), 3)

        party.send_params_to(1, k0)
        party.send_params_to(2, k1)

        party.receive_torch_tensor_from(1)
        party.receive_torch_tensor_from(2)

        out = RingTensor.zeros(z.shape, z.replicated_shared_tensor[0].dtype, z.replicated_shared_tensor[0].scale)

    else:
        if GE_TYPE == 'FSS':
            k = DICFKey.from_dic(party.receive_params_dict_from(0))
        else:
            k = CMPSigmaKey.from_dic(party.receive_params_dict_from(0))
        party.send_torch_tensor_to(0, torch.tensor(1))

        if DEBUG:
            if GE_TYPE == 'FSS':
                r_in = [k.r, RingTensor.convert_to_ring(0)]  # DICF
            else:
                r_in = [k.r_in, RingTensor.convert_to_ring(0)]  # CMPSigma
        else:
            if GE_TYPE == 'FSS':
                r_in = [k.r.reshape(z.shape), RingTensor.convert_to_ring(0)]
            else:
                r_in = [k.r_in.reshape(z.shape), RingTensor.convert_to_ring(0)]  # CMPSigma
        r_in = list_rotate(r_in, party.party_id - 1)

        r = z + ReplicatedSecretSharing(r_in, party)

        party.send_ring_tensor_to(party.party_id % 2 + 1, r.replicated_shared_tensor[party.party_id - 1])
        r_other = party.receive_ring_tensor_from(party.party_id % 2 + 1)

        z_shift = r.replicated_shared_tensor[0] + r.replicated_shared_tensor[1] + r_other

        if GE_TYPE == 'FSS':
            out = DICF.eval(z_shift, k, party.party_id - 1)
        else:
            out = CMPSigma.eval(z_shift, k, party.party_id - 1)

    rand = ReplicatedSecretSharing.rand_like(z, party)

    out = out ^ rand.replicated_shared_tensor[0].get_bit(1) ^ rand.replicated_shared_tensor[1].get_bit(1)

    party.send_ring_tensor_to((party.party_id + 2) % 3, out)
    out_other = party.receive_ring_tensor_from((party.party_id + 1) % 3)

    out = ReplicatedSecretSharing([out, out_other], party)

    return bit_injection(out) * x.replicated_shared_tensor[0].scale


def b2a(x: ReplicatedSecretSharing):
    pass


def bit_injection(x: ReplicatedSecretSharing):
    """
    单bit位b2a协议
    参考ABY3实现，令P2为sender，P1为receiver，P0为helper

    :param x:
    :return:
    """
    party = x.party
    if party.party_id == 0:
        c0 = RingTensor.random(x.shape, dtype=x.replicated_shared_tensor[0].dtype,
                               scale=x.replicated_shared_tensor[0].scale, device=x.device)
        party.send_ring_tensor_to(2, c0)

        OT.helper(x.replicated_shared_tensor[1], party, 2, 1)

        c1 = party.receive_ring_tensor_from(1)

        return ReplicatedSecretSharing([c0, c1], party)

    if party.party_id == 1:
        c2 = RingTensor.random(x.shape, dtype=x.replicated_shared_tensor[0].dtype,
                               scale=x.replicated_shared_tensor[0].scale, device=x.device)
        party.send_ring_tensor_to(2, c2)

        c1 = OT.receiver(x.replicated_shared_tensor[0], party, 2, 0)
        c1 = RingTensor(c1, dtype=x.replicated_shared_tensor[0].dtype,
                        scale=x.replicated_shared_tensor[0].scale, device=x.device)
        party.send_ring_tensor_to(0, c1)

        return ReplicatedSecretSharing([c1, c2], party)

    if party.party_id == 2:
        c0 = party.receive_ring_tensor_from(0)
        c2 = party.receive_ring_tensor_from(1)

        m0 = (x.replicated_shared_tensor[0] ^ x.replicated_shared_tensor[1]) - c0 - c2
        m1 = (x.replicated_shared_tensor[0] ^ x.replicated_shared_tensor[1] ^ 1) - c0 - c2

        OT.sender(m0.tensor, m1.tensor, party, 1, 0)

        return ReplicatedSecretSharing([c2, c0], party)


def truncate(share: ReplicatedSecretSharing, scale=SCALE):
    # TODO: truncate有问题
    # r, r_t = truncate_preprocess(share, scale)
    # delta_share = share - r
    delta = share.restore()
    return share

    # return r_t + delta // scale


def truncate_preprocess(share: ReplicatedSecretSharing, scale=SCALE):
    # TODO: 后续是否要改成离线操作
    r_share = RingTensor.random(share.shape, dtype=share.replicated_shared_tensor[0].dtype,
                                scale=share.replicated_shared_tensor[0].scale, device=share.device)
    r_share = r_share // 3  # 把随机数的范围缩小减少错误概率 TODO: 此处可能会带来额外的安全问题
    r_t_share = r_share // scale

    share.party.send_ring_tensor_to((share.party.party_id - 1) % 3, r_share)
    r_share_other = share.party.receive_ring_tensor_from((share.party.party_id + 1) % 3)

    r_t_share_other = r_share_other // scale

    r_share = ReplicatedSecretSharing([r_share, r_share_other], share.party)
    r_t_share = ReplicatedSecretSharing([r_t_share, r_t_share_other], share.party)

    if share.party.party_id == 0:
        r_2 = RingTensor.random(share.shape, dtype=share.replicated_shared_tensor[0].dtype,
                                scale=share.replicated_shared_tensor[0].scale, device=share.device)
        r_2_share = ReplicatedSecretSharing.gen_and_share(r_2, share.party)
        share.party.send_ring_tensor_to(1, r_2)

        r_t_2 = RingTensor.random(share.shape, dtype=share.replicated_shared_tensor[0].dtype,
                                  scale=share.replicated_shared_tensor[0].scale, device=share.device) // scale
        r_t_2_share = ReplicatedSecretSharing.gen_and_share(r_t_2, share.party)
        share.party.send_ring_tensor_to(1, r_t_2)

        r_3_share = share.party.receive_rss_from(1)
        r_t_3_share = share.party.receive_rss_from(1)

        r_1_share = r_share - r_2_share - r_3_share
        r_t_1_share = r_t_share - r_t_2_share - r_t_3_share

        share.party.send_rss_to(2, r_1_share)
        r_1_share_2 = share.party.receive_rss_from(2)

        r_1 = r_1_share.replicated_shared_tensor[0] + r_1_share.replicated_shared_tensor[1] \
              + r_1_share_2.replicated_shared_tensor[0]

        share.party.send_rss_to(2, r_t_1_share)
        r_t_1_share_2 = share.party.receive_rss_from(2)

        r_t_1 = r_t_1_share.replicated_shared_tensor[0] + r_t_1_share.replicated_shared_tensor[1] \
                + r_t_1_share_2.replicated_shared_tensor[0]

        r = ReplicatedSecretSharing([r_1, r_2], share.party)
        r_t = ReplicatedSecretSharing([r_t_1, r_t_2], share.party)

        return r, r_t

    if share.party.party_id == 1:
        r_2_share = share.party.receive_rss_from(0)
        r_2 = share.party.receive_ring_tensor_from(0)

        r_t_2_share = share.party.receive_rss_from(0)
        r_t_2 = share.party.receive_ring_tensor_from(0)

        r_3 = RingTensor.random(share.shape, dtype=share.replicated_shared_tensor[0].dtype,
                                scale=share.replicated_shared_tensor[0].scale, device=share.device)
        r_3_share = ReplicatedSecretSharing.gen_and_share(r_3, share.party)
        share.party.send_ring_tensor_to(2, r_3)

        r_t_3 = RingTensor.random(share.shape, dtype=share.replicated_shared_tensor[0].dtype,
                                  scale=share.replicated_shared_tensor[0].scale, device=share.device) // scale
        r_t_3_share = ReplicatedSecretSharing.gen_and_share(r_t_3, share.party)
        share.party.send_ring_tensor_to(2, r_t_3)

        r = ReplicatedSecretSharing([r_2, r_3], share.party)
        r_t = ReplicatedSecretSharing([r_t_2, r_t_3], share.party)

        return r, r_t

    if share.party.party_id == 2:
        r_2_share = share.party.receive_rss_from(0)
        r_t_2_share = share.party.receive_rss_from(0)

        r_3_share = share.party.receive_rss_from(1)
        r_3 = share.party.receive_ring_tensor_from(1)

        r_t_3_share = share.party.receive_rss_from(1)
        r_t_3 = share.party.receive_ring_tensor_from(1)

        r_1_share = r_share - r_2_share - r_3_share
        r_t_1_share = r_t_share - r_t_2_share - r_t_3_share

        r_1_share_0 = share.party.receive_rss_from(0)
        share.party.send_rss_to(0, r_1_share)

        r_1 = r_1_share.replicated_shared_tensor[0] + r_1_share.replicated_shared_tensor[1] \
              + r_1_share_0.replicated_shared_tensor[1]

        r_t_1_share_0 = share.party.receive_rss_from(0)
        share.party.send_rss_to(0, r_t_1_share)

        r_t_1 = r_t_1_share.replicated_shared_tensor[0] + r_t_1_share.replicated_shared_tensor[1] \
                + r_t_1_share_0.replicated_shared_tensor[1]

        r = ReplicatedSecretSharing([r_3, r_1], share.party)
        r_t = ReplicatedSecretSharing([r_t_3, r_t_1], share.party)

        return r, r_t

    else:
        raise ValueError("party id error")
