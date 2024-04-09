"""
Exp(x) = exp(x)
x的有效范围是(-16, 21.5)
"""
import torch

from config.base_configs import data_type, DEVICE
from crypto.protocols.arithmetic_secret_sharing.arithmetic_secret_sharing import b2a
from crypto.primitives.auxiliary_parameter.parameter import Parameter
from crypto.protocols.comparison.cmp_sigma import CMPSigmaKey, CMPSigma
from crypto.protocols.look_up.look_up_table import LookUpKey, LookUp
from crypto.tensor.ArithmeticSharedRingTensor import ArithmeticSharedRingTensor
from crypto.tensor.RingTensor import RingTensor


class Exp(object):
    @staticmethod
    def eval(x: ArithmeticSharedRingTensor, key, scale_bit=16):
        return exp_eval(x, key, scale_bit)


class ExpKey(Parameter):
    def __init__(self):
        self.high_look_up_key = LookUpKey()  # 高位查考表的密钥
        self.low_look_up_key = LookUpKey()  # 低位查考表的密钥
        self.high_table = None  # 高位查找表
        self.low_table = None  # 低位查找表
        self.sigma_key = None

    def __getitem__(self, item):
        key = super(ExpKey, self).__getitem__(item)
        key.high_table = self.high_table
        key.low_table = self.low_table
        return key

    @staticmethod
    def gen(num_of_keys, scale_bit=16):
        return exp_gen(num_of_keys, scale_bit)


def exp_gen(num_of_keys, scale_bit=16):
    pos_table_bit = (scale_bit + 6) // 2
    neg_table_bit = (scale_bit + 4) // 2

    k0, k1 = ExpKey(), ExpKey()

    k0.high_look_up_key, k1.high_look_up_key = LookUpKey.gen(num_of_keys, 0, 2 ** pos_table_bit + 2 ** neg_table_bit)
    k0.low_look_up_key, k1.low_look_up_key = LookUpKey.gen(num_of_keys, 0, 2 ** pos_table_bit + 2 ** neg_table_bit)

    k0.sigma_key, k1.sigma_key = CMPSigmaKey.gen(num_of_keys)

    high_table, low_table = create_table(scale_bit)

    k0.high_table = k1.high_table = high_table
    k0.low_table = k1.low_table = low_table

    return k0, k1


def create_table(scale_bit=16):
    pos_table_bit = (scale_bit + 6) // 2
    neg_table_bit = (scale_bit + 4) // 2
    each_table_bit = (scale_bit + 10) // 2

    pos_i = torch.arange(0, 2 ** pos_table_bit, dtype=data_type, device=DEVICE)
    high_table_pos = RingTensor.convert_to_ring(torch.exp(pos_i / (2 ** (scale_bit - each_table_bit))))
    low_table_pos = RingTensor.convert_to_ring(torch.exp(pos_i / (2 ** scale_bit)))

    neg_i = torch.arange(- 2 ** neg_table_bit, 0, dtype=data_type, device=DEVICE)
    high_table_neg = RingTensor.convert_to_ring(torch.exp(neg_i / (2 ** (scale_bit - each_table_bit))))
    low_table_neg = RingTensor.convert_to_ring(torch.exp(neg_i / (2 ** scale_bit)))

    high_table = RingTensor.cat((high_table_pos, high_table_neg), dim=0)
    low_table = RingTensor.cat((low_table_pos, low_table_neg), dim=0)

    return high_table, low_table


def exp_eval(x, key, scale_bit=16):
    pos_table_bit = (scale_bit + 6) // 2
    neg_table_bit = (scale_bit + 4) // 2

    low_bound = -2 ** (scale_bit + 4)
    upper_bound = 2 ** (scale_bit + 6)
    shape = x.shape

    high_table = key.high_table
    low_table = key.low_table

    sigma_key = key.sigma_key

    if isinstance(sigma_key, dict):
        sigma_key = CMPSigmaKey.from_dic(sigma_key)

    x_shift = ArithmeticSharedRingTensor(sigma_key.r_in, x.party) + x.flatten()
    x_shift = x_shift.restore()

    i = CMPSigma.eval(x_shift - low_bound, sigma_key, x.party.party_id) ^ CMPSigma.eval(x_shift - upper_bound,
                                                                                        sigma_key, x.party.party_id)

    i = b2a(i.tensor, x.party)

    i.dtype = x.dtype
    i = i * x.scale

    c = i * x.flatten() - (i - RingTensor.ones_like(i)) * low_bound

    # Look Up Table
    high = c / (2 ** ((scale_bit + 10) // 2))
    low = c - high * (2 ** ((scale_bit + 10) // 2))

    t_high = LookUp.eval(high, key.high_look_up_key, 0, 2 ** pos_table_bit + 2 ** neg_table_bit, high_table)
    t_low = LookUp.eval(low, key.low_look_up_key, 0, 2 ** pos_table_bit + 2 ** neg_table_bit, low_table)

    return (t_high * t_low).reshape(shape)
