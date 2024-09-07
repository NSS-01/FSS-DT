"""
算术秘密共享
"""
import math
from functools import singledispatchmethod

from common.utils.cuda_utils import cuda_matmul
from crypto.protocols.arithmetic_secret_sharing.arithmetic_secret_sharing import *
from crypto.tensor.RingTensor import RingTensor


def override_with_party(method_names):
    def decorator(cls):
        def override(method_name):
            def delegate(self, *args, **kwargs):
                result = getattr(super(cls, self), method_name)(*args, **kwargs)
                result.party = self.party
                return result

            return delegate

        for name in method_names:
            setattr(cls, name, override(name))
        return cls

    return decorator


@override_with_party(
    ['__getitem__', '__neg__', 'reshape', 'view', 'transpose', 'squeeze', 'unsqueeze', 'flatten', 'clone', 'pad', 'sum',
     'repeat', 'permute'])
class ArithmeticSharedRingTensor(RingTensor):
    """
    支持算术秘密共享运算的自定义tensor类

    属性：
        party: 所属参与方
        ring_tensor: 其张量值，RingTensor类型
        shape: tensor的shape
        share_type: 支持的共享类型，支持22秘密共享
        device: 运算所在设备
    """

    @singledispatchmethod
    def __init__(self):
        super().__init__(self)
        self.party = None

    @__init__.register(torch.Tensor)
    def _from_tensor(self, tensor=torch.tensor(0), dtype='int', device=DEVICE, party=None):
        super(ArithmeticSharedRingTensor, self).__init__(tensor, dtype, device)
        self.party = party

    @__init__.register(RingTensor)
    def _from_ring_tensor(self, tensor: RingTensor, party):
        super(ArithmeticSharedRingTensor, self).__init__(tensor.tensor, tensor.dtype, tensor.device)
        self.party = party

    @property
    def ring_tensor(self):
        return RingTensor(self.tensor, self.dtype, self.device)

    @property
    def T(self):
        result = super().T
        result.party = self.party
        return result

    def __getstate__(self):
        state = self.__dict__.copy()
        state['party'] = None
        return state

    def __str__(self):
        return "{}\n party:{}".format(super(ArithmeticSharedRingTensor, self).__str__(), self.party.party_id)

    def __add__(self, other):
        if isinstance(other, ArithmeticSharedRingTensor):
            new_tensor = self.tensor + other.tensor
        elif isinstance(other, RingTensor):  # for ring tensor, only party 0 add it to the share tensor
            if self.party.party_id == 0:
                new_tensor = self.tensor + other.tensor
            else:
                new_tensor = self.tensor
        else:
            raise TypeError(f"unsupported operand type(s) for + '{type(self)}' and {type(other)}")
        return ArithmeticSharedRingTensor(new_tensor, self.dtype, self.device, self.party)

    def __sub__(self, other):
        if isinstance(other, ArithmeticSharedRingTensor):
            new_tensor = self.tensor - other.tensor
        elif isinstance(other, RingTensor):
            if self.party.party_id == 0:
                new_tensor = self.tensor - other.tensor
            else:
                new_tensor = self.tensor
        else:
            raise TypeError(f"unsupported operand type(s) for - '{type(self)}' and {type(other)}")
        return ArithmeticSharedRingTensor(new_tensor, self.dtype, self.device, self.party)

    def __mul__(self, other):
        if isinstance(other, ArithmeticSharedRingTensor):
            res = beaver_mul(self, other)
        elif isinstance(other, RingTensor):
            res = ArithmeticSharedRingTensor(self.tensor * other.tensor, self.dtype, self.device, self.party)
        elif isinstance(other, int):
            return ArithmeticSharedRingTensor(self.tensor * other, self.dtype, self.device, self.party)
        else:
            raise TypeError(f"unsupported operand type(s) for * '{type(self)}' and {type(other)}")
        if res.dtype == 'float':
            res = res / self.scale

        return res

    def __matmul__(self, other):
        if isinstance(other, ArithmeticSharedRingTensor):
            res = secure_matmul(self, other)
        elif isinstance(other, RingTensor):
            res = ArithmeticSharedRingTensor(cuda_matmul(self.tensor, other.tensor), self.dtype, self.device,
                                             self.party)
        else:
            raise TypeError(f"unsupported operand type(s) for @ '{type(self)}' and {type(other)}")
        if res.dtype == 'float':
            res = res / self.scale

        return res

    def __pow__(self, power, modulo=None):
        """
        ASS的幂函数, ' continue' coming soon TODO: 快速幂？
        :param power:幂次数
        :param modulo:计算模式，有'mul'和 'continu'
        :return:
        """
        if isinstance(power, int):
            temp = self
            res = temp
            for i in range(1, power):
                res = res * temp
            return res
        else:
            raise TypeError(f"unsupported operand type(s) for ** '{type(self)}' and {type(power)}")

    def __truediv__(self, other):
        if isinstance(other, int):
            temp = self.clone()
            if other == 2:  # TODO: 使用别的truncate方法或可省略这一步
                return temp * RingTensor.convert_to_ring(torch.tensor(0.5, device=temp.device))
            return truncate(temp, other)
        elif isinstance(other, float):
            return (self * 65536) / int(other * 65536)
        elif isinstance(other, ArithmeticSharedRingTensor):
            return secure_div(self, other)
        elif isinstance(other, RingTensor):
            return truncate(self * other.scale, other.tensor)
        else:
            raise TypeError(f"unsupported operand type(s) for / '{type(self)}' and {type(other)}")

    def __mod__(self, other):
        if isinstance(other, int):
            return ArithmeticSharedRingTensor(self.tensor % other, self.dtype, self.device, self.party)
        else:
            raise TypeError(f"unsupported operand type(s) for % '{type(self)}' and {type(other)}")

    def __le__(self, other):
        if isinstance(other, ArithmeticSharedRingTensor):
            return secure_ge(other, self)
        else:
            raise TypeError(f"unsupported operand type(s) for comparison '{type(self)}' and {type(other)}")

    def __ge__(self, other):
        if isinstance(other, ArithmeticSharedRingTensor):

            return secure_ge(self, other)
        else:
            raise TypeError(f"unsupported operand type(s) for comparison '{type(self)}' and {type(other)}")

    def __lt__(self, other):
        if isinstance(other, ArithmeticSharedRingTensor):
            ge = secure_ge(self, other)
            return (ge - RingTensor.ones_like(ge)) * -1
        else:
            raise TypeError(f"unsupported operand type(s) for comparison '{type(self)}' and {type(other)}")

    def __gt__(self, other):
        if isinstance(other, ArithmeticSharedRingTensor):
            ge = secure_ge(other, self)
            return (ge - RingTensor.ones_like(ge)) * -1
        else:
            raise TypeError(f"unsupported operand type(s) for comparison '{type(self)}' and {type(other)}")

    @classmethod
    def cat(cls, tensors, dim=0, party=None):
        result = super().cat(tensors)
        if party is None:
            party = tensors[0].party
        result.party = party
        return result

    @classmethod
    def load_from_file(cls, file_path, party=None):
        result = super().load_from_file(file_path)
        result.party = party
        return result

    @classmethod
    def empty(cls, size, dtype='int', device=DEVICE, party=None):
        result = super().empty(size, dtype, device)
        result.party = party
        return result

    @classmethod
    def empty_like(cls, tensor, party=None):  # TODO
        result = super().empty_like(tensor)
        result.party = party
        return result

    @classmethod
    def roll(cls, input, shifts, dims=0, party=None):
        result = super().roll(input, shifts, dims)
        result.party = party
        return result

    @classmethod
    def row_shift(cls, input, shifts, party=None):
        result = super().row_shift(input, shifts)
        result.party = party
        return result

    @classmethod
    def nexp(cls, x):
        return secure_neg_exp(x, int(math.log2(x.scale)))

    @classmethod
    def exp(cls, x):
        return secure_exp(x, int(math.log2(x.scale)))
        # return secure_exp_by_lim(x)

    @classmethod
    def reciprocal_sqrt(cls, x):
        return secure_reciprocal_sqrt(x)

    @classmethod
    def max(cls, x, dim=None):
        def max_(inputs):
            if inputs.shape[0] == 1:
                return inputs
            if inputs.shape[0] % 2 == 1:
                inputs_ = inputs[-1:]
                inputs = ArithmeticSharedRingTensor.cat([inputs, inputs_], 0)
            inputs_0 = inputs[0::2]
            inputs_1 = inputs[1::2]

            ge = inputs_0 >= inputs_1
            le = (ge - RingTensor.ones_like(ge)) * -1

            return ge * inputs_0 + le * inputs_1

        if dim is None:
            x = x.flatten()
        else:
            x = x.transpose(dim, 0)
        if x.shape[0] == 1:
            return x.transpose(dim, 0).squeeze(-1)
        else:
            x = max_(x)
        return ArithmeticSharedRingTensor.max(x.transpose(0, dim), dim)

    @staticmethod
    def restore_two_shares(share_0, share_1):
        """
        静态方法，通过两个分享值恢复原数据

        :param share_0: 份额Xa
        :param share_1: 份额Xb
        :return: 还原后的明文值X
        """
        return share_0.ring_tensor + share_1.ring_tensor

    def restore(self):
        from crypto.mpc.semi_honest_party import SemiHonestCS
        from crypto.mpc.semi_honest_party import SemiHonestMPCParty
        if isinstance(self.party, SemiHonestMPCParty):
            sum = self
            for i in range(self.party.parties_num):
                self.party.send_share_to((self.party.party_id + i) % self.party.parties_num, self)

            for i in range(self.party.parties_num):
                other = self.party.receive_shares_from((self.party.party_id + i) % self.party.parties_num)
                sum += other
            return sum
        elif isinstance(self.party, SemiHonestCS):
            # send shares to other parties
            self.party.send(self)
            # receive shares from other parties
            other = self.party.receive()
            return self.ring_tensor + other.ring_tensor
    def restore_to(self, c):
        from crypto.mpc.semi_honest_party import SemiHonestCS
        if isinstance(self.party,SemiHonestCS):
            if self.party.party_id == c:
                other = self.party.receive()
            else:
                self.party.send(self)
            return  self.ring_tensor + other.ring_tensor
    @staticmethod
    def share(tensor: RingTensor, num_of_party: int):
        """
        对一个环上的张量(RingTensor)进行加法秘密共享
        静态方法，用于记录具体的秘密共享，即用一个列表将每个共享的值都存储下来

        :param tensor: 要进行共享操作的张量
        :param num_of_party: 参与分享的参与方数量
        :return: 存放各秘密份额的列表(List),元素类型仍为RingTensor
        """
        share = []
        x_0 = tensor.clone()

        for i in range(num_of_party - 1):
            x_i = RingTensor.random(tensor.shape, dtype=tensor.dtype, device=tensor.device)
            share.append(x_i)
            x_0 -= x_i
        share.append(x_0)
        return share

    @staticmethod
    def share_to_other(tensor: RingTensor, party):
        tensor_0 = RingTensor.random(tensor.shape, dtype=tensor.dtype)
        tensor_1 = tensor - tensor_0
        shared_tensor = ArithmeticSharedRingTensor(tensor_1, party)
        party.send(tensor_0)
        return shared_tensor

    @staticmethod
    def receive_share(party):
        tensor_0 = party.receive()
        shared_tensor = ArithmeticSharedRingTensor(tensor_0, party)
        return shared_tensor
