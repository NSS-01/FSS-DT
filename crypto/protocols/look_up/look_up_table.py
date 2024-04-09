import math

from config.base_configs import DEBUG
from crypto.primitives.auxiliary_parameter.parameter import Parameter
from crypto.primitives.function_secret_sharing.dpf import DPFKey, DPF
from crypto.tensor.ArithmeticSharedRingTensor import ArithmeticSharedRingTensor
from crypto.tensor.RingTensor import RingTensor


class LookUp(object):
    @staticmethod
    def eval(x: ArithmeticSharedRingTensor, key, down, upper, table: RingTensor):
        return look_up_eval(x, key, down, upper, table)


class LookUpKey(Parameter):
    def __init__(self):
        self.dpf_value = None
        self.down_bound = None
        self.upper_bound = None
        self.phi = None

    @staticmethod
    def gen(num_of_keys, down, upper):
        return look_up_gen(num_of_keys, down, upper)


def look_up_gen(num_of_keys, down, upper):
    # TODO: 目前仅支持num_of_keys=1的情况，即debug模式
    phi = RingTensor.random([num_of_keys], down_bound=down, upper_bound=upper)
    phi.bit_len = math.ceil(math.log2(upper - down))

    k0 = LookUpKey()
    k1 = LookUpKey()

    dpf_k0, dpf_k1 = DPFKey.gen(num_of_keys, phi, RingTensor.convert_to_ring(1))
    i = RingTensor.arange(start=down, end=upper, device=phi.device)
    i = i.repeat(num_of_keys, 1)
    i.bit_len = math.ceil(math.log2(upper - down))

    k0.dpf_value = DPF.eval(i, dpf_k0, 0)
    k1.dpf_value = DPF.eval(i, dpf_k1, 1)

    k0.phi, k1.phi = ArithmeticSharedRingTensor.share(phi, 2)
    k0.down_bound = k1.down_bound = down
    k0.upper_bound = k1.upper_bound = upper

    return k0, k1


def look_up_eval(x: ArithmeticSharedRingTensor, key: LookUpKey, down, upper, table: RingTensor):
    # assert key.down_bound == down and key.upper_bound == upper, "The intervals do not match"
    shape = x.shape
    x = x.flatten()
    x_shift_shared = ArithmeticSharedRingTensor(key.phi, x.party) - x
    x_shift = x_shift_shared.restore()

    # i = RingTensor.arange(start=down, end=upper, dtype=x_shift.dtype, device=x_shift.device)
    # i = i.repeat(x.shape[0], 1)
    # i.bit_len = math.ceil(math.log2(upper - down))

    import torch

    y = RingTensor(key.dpf_value, x_shift.dtype, x_shift.device)
    if DEBUG:
        y = y.repeat(x.shape[0], 1)
    u = ArithmeticSharedRingTensor.row_shift(y, shifts=x_shift.tensor, party=x.party)
    res = (u * table).sum(-1)
    res.dtype = x.dtype

    return res.reshape(shape)


def gen_for_exp():
    pass


def gen_for_layer_norm():
    pass
