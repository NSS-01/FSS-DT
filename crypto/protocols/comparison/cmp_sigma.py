import torch

from config.base_configs import DEBUG, DEVICE, HALF_RING
from crypto.tensor.ArithmeticSharedRingTensor import ArithmeticSharedRingTensor
from crypto.primitives.auxiliary_parameter.parameter import Parameter
from crypto.primitives.function_secret_sharing.dpf import DPFKey
from crypto.primitives.function_secret_sharing.p_dicf import pps_calculate
from crypto.tensor.RingTensor import RingTensor


class CMPSigma(object):
    @staticmethod
    def eval(x_shift: RingTensor, keys, party_id):
        return cmp_sigma_eval(party_id, keys, x_shift)


class CMPSigmaKey(Parameter):
    def __init__(self):
        self.dpf_key = DPFKey()
        self.c = None
        self.r_in = None
        self.size = 0

    @staticmethod
    def gen(num_of_keys):
        return cmp_sigma_gen(num_of_keys)


def cmp_sigma_gen(num_of_keys):
    k0 = CMPSigmaKey()
    k1 = CMPSigmaKey()

    r_in = RingTensor.random([num_of_keys])
    x1 = r_in
    y1 = r_in % (HALF_RING - 1)
    k0.dpf_key, k1.dpf_key = DPFKey.gen(num_of_keys, y1, RingTensor.convert_to_ring(1))
    c = x1.signbit()
    c0 = torch.randint(0, 1, [num_of_keys], device=DEVICE)
    c0 = RingTensor.convert_to_ring(c0)
    c1 = c ^ c0

    k0.c = c0
    k1.c = c1

    k0.r_in, k1.r_in = ArithmeticSharedRingTensor.share(r_in, 2)

    return k0, k1


def cmp_sigma_eval(party_id, key, x_shift: RingTensor):
    shape = x_shift.shape
    x_shift = x_shift.view(-1, 1)
    y = x_shift % (HALF_RING - 1)
    y = y + 1
    out = pps_calculate(y, key.dpf_key, party_id)
    out = x_shift.signbit() * party_id ^ key.c.view(-1, 1) ^ out
    return out.view(shape)
