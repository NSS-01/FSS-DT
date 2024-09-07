"""
nExp(x) = exp(-x)
x的有效输入范围是[0, 16)
"""
import torch

from config.base_configs import data_type, DEVICE
from crypto.primitives.auxiliary_parameter.parameter import Parameter
from crypto.protocols.look_up.look_up_table import LookUpKey, LookUp
from crypto.tensor.ArithmeticSharedRingTensor import ArithmeticSharedRingTensor
from crypto.tensor.RingTensor import RingTensor


class NegExp(object):
    @staticmethod
    def eval(x: ArithmeticSharedRingTensor, key, scale_bit=16):
        return neg_exp_eval(x, key, scale_bit)


class NegExpKey(Parameter):
    def __init__(self):
        self.high_look_up_key = LookUpKey()  # 高位查考表的密钥
        self.low_look_up_key = LookUpKey()  # 低位查考表的密钥
        self.high_table = None  # 高位查找表
        self.low_table = None  # 低位查找表

    def __getitem__(self, item):
        key = super(NegExpKey, self).__getitem__(item)
        key.high_table = self.high_table
        key.low_table = self.low_table
        return key

    @staticmethod
    def gen(num_of_keys, scale_bit=16):
        return neg_exp_gen(num_of_keys, scale_bit)


def neg_exp_gen(num_of_keys, scale_bit=16):
    each_table_bit = (scale_bit + 4) // 2

    k0, k1 = NegExpKey(), NegExpKey()

    k0.high_look_up_key, k1.high_look_up_key = LookUpKey.gen(num_of_keys, 0, 2 ** each_table_bit)
    k0.low_look_up_key, k1.low_look_up_key = LookUpKey.gen(num_of_keys, 0, 2 ** each_table_bit)

    high_table, low_table = create_table(scale_bit)

    k0.high_table = k1.high_table = high_table
    k0.low_table = k1.low_table = low_table

    return k0, k1


def create_table(scale_bit=16):
    each_table_bit = (scale_bit + 4) // 2
    i = torch.arange(0, 2 ** each_table_bit, dtype=data_type, device=DEVICE)
    high_table = RingTensor.convert_to_ring(torch.exp(-(i / (2 ** (scale_bit - each_table_bit)))))
    low_table = RingTensor.convert_to_ring(torch.exp(-(i / (2 ** scale_bit))))

    return high_table, low_table


def neg_exp_eval(x, key, scale_bit=16):
    total_bit = scale_bit + 4
    shape = x.shape

    high_table = key.high_table
    low_table = key.low_table

    ge = x >= ArithmeticSharedRingTensor(RingTensor(2 ** (total_bit - 1), dtype=x.dtype, device=x.device), x.party)
    lt = (ge - RingTensor.ones_like(ge)) * -1

    c = lt * x + ge * ((2 ** total_bit - 1) // x.scale)

    # Look Up Table
    high = c / (2 ** (total_bit // 2))
    low = c - high * (2 ** (total_bit // 2))

    t_high = LookUp.eval(high, key.high_look_up_key, 0, 2 ** (total_bit // 2), high_table)
    t_low = LookUp.eval(low, key.low_look_up_key, 0, 2 ** (total_bit // 2), low_table)

    return (t_high * t_low).reshape(shape)
