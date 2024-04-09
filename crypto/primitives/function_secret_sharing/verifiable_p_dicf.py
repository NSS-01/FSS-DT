import torch

from config.base_configs import LAMBDA, DEVICE, data_type, PRG_TYPE, HALF_RING, DEBUG
from common.random.prg import PRG
from crypto.tensor.RingTensor import RingTensor
from crypto.primitives.function_secret_sharing.verifiable_dpf import vdpf_gen, ver_ppq_dpf


class vpDICF(object):
    @staticmethod
    def gen(num_of_keys):
        return vp_dicf_gen(num_of_keys)

    @staticmethod
    def eval(x_shift, key, lower_bound, upper_bound, party_id):
        return vp_dicf_eval(x_shift, key, lower_bound, upper_bound, party_id)

    @staticmethod
    def cmp(x, key, party_id):
        lower_bound = torch.tensor(0)
        upper_bound = torch.tensor(HALF_RING - 1)
        return vp_dicf_eval(x, key, lower_bound, upper_bound, party_id)


def vp_dicf_gen(num_of_keys):
    phi = RingTensor.random([num_of_keys])
    # beta is fix to 1
    beta = RingTensor.convert_to_ring(1)
    K0, K1 = vdpf_gen(num_of_keys, phi, beta)
    return phi, K0, K1


def vp_dicf_eval(x_shift, key, lower_bound, upper_bound, party_id):
    p = RingTensor.convert_to_ring(lower_bound + x_shift.tensor)
    q = RingTensor.convert_to_ring(upper_bound - x_shift.tensor)

    cond = (p.tensor ^ q.tensor) < 0
    tau = ((p > q) ^ cond) * party_id

    parity_q, pi_q = ver_ppq_dpf(q, key, party_id)
    parity_p, pi_p = ver_ppq_dpf(p, key, party_id)
    ans = parity_p ^ parity_q ^ tau
    pi = pi_p ^ pi_q

    return ans, pi
