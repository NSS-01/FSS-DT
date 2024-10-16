import math

from NssMPC import RingTensor, ArithmeticSecretSharing
from NssMPC.crypto.aux_parameter import Parameter, DPFKey
from NssMPC.crypto.primitives import DPF


class LookUpKey(Parameter):
    def __init__(self):
        self.dpf_key = DPFKey()
        self.down_bound = None
        self.upper_bound = None
        self.phi = None

    @staticmethod
    def gen(num_of_keys, down, upper):
        return look_up_gen(num_of_keys, down, upper)


def look_up_gen(num_of_keys, down, upper):
    phi = RingTensor.random([num_of_keys], down_bound=down, upper_bound=upper)
    phi.bit_len = math.ceil(math.log2(upper - down))

    k0 = LookUpKey()
    k1 = LookUpKey()

    k0.dpf_key, k1.dpf_key = DPFKey.gen(num_of_keys, phi, RingTensor.convert_to_ring(1))
    k0.phi, k1.phi = ArithmeticSecretSharing.share(phi, 2)
    k0.down_bound = k1.down_bound = down
    k0.upper_bound = k1.upper_bound = upper

    return k0, k1


def look_up_eval(x, key: LookUpKey, table: RingTensor):
    down = key.down_bound
    upper = key.upper_bound
    shape = x.shape
    x = x.flatten()
    key.phi.party = x.party
    x_shift_shared = key.phi - x
    x_shift = x_shift_shared.restore()

    i = RingTensor.arange(start=down, end=upper, dtype=x_shift.dtype, device=x_shift.device)
    i = i.repeat(x.shape[0], 1)
    i.bit_len = math.ceil(math.log2(upper - down))

    y = DPF.eval(i, key.dpf_key, x.party.party_id)
    y = ArithmeticSecretSharing(y, x.party)

    u = ArithmeticSecretSharing.rotate(y, shifts=-x_shift)
    res = (u * table).sum(-1)
    res.dtype = x.dtype

    return res.reshape(shape)
