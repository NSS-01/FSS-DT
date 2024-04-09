from crypto.primitives.function_secret_sharing.verifiable_dpf import VerifiableDPF, VerifiableDPFKey
from crypto.tensor.ArithmeticSharedRingTensor import ArithmeticSharedRingTensor

from crypto.tensor.RingTensor import RingTensor
from config.base_configs import *


class VerSigma(object):
    @staticmethod
    def gen(num_of_keys):
        return verifiable_sigma_gen(num_of_keys)

    @staticmethod
    def eval(x_shift: RingTensor, keys, party_id):
        return verifiable_sigma_eval(party_id, keys, x_shift)

    @staticmethod
    def cmp_eval(x: ArithmeticSharedRingTensor, keys, party_id):
        if DEBUG:
            x_shift = ArithmeticSharedRingTensor(keys.r_in, x.party) + x
        else:
            x_shift = ArithmeticSharedRingTensor(keys.r_in.reshape(x.shape), x.party) + x
        x_shift = x_shift.restore()
        return verifiable_sigma_eval(party_id, keys, x_shift)


class VerSigmaKey(object):
    def __init__(self, ver_dpf_key):
        self.ver_dpf_key = ver_dpf_key
        self.c = None
        self.r_in = None
        self.size = 0

    def __getitem__(self, item):
        k = VerSigmaKey(self.ver_dpf_key[item])
        k.c = self.c[item]
        k.r_in = self.r_in[item]
        return k

    def __setitem__(self, key, value):
        self.ver_dpf_key[key] = value.ver_dpf_key
        self.c[key] = value.c
        self.r_in[key] = value.r_in

    def to_dic(self):
        """
        将CMPSigmaKey类对象转换为字典
        :return: 表示类对象的字典
        """
        dict = self.ver_dpf_key.to_dic()
        dict['c'] = self.c
        dict['r_in'] = self.r_in
        return dict

    @staticmethod
    def dic_to_key(dic):
        """
        将字典转换为类对象
        :param dic: 字典对象
        :return: CMPSigmaKey对象
        """
        key = VerSigmaKey(VerifiableDPFKey.dic_to_key(dic))
        key.c = dic['c']
        key.r_in = dic['r_in']

        if key.r_in.shape == torch.Size([]):
            key.size = 1
        else:
            key.size = key.r_in.shape[0]
        return key

    def to(self, device):
        self.ver_dpf_key = self.ver_dpf_key.to(device)
        self.c = self.c.to(device)
        self.r_in = self.r_in.to(device)
        return self

    @staticmethod
    def empty(size):
        key = VerSigmaKey(VerifiableDPFKey.empty(size))
        key.c = torch.empty(size, dtype=torch.bool, device=DEVICE)
        key.size = size
        return key

    @staticmethod
    def empty_like(key):
        key = VerSigmaKey(VerifiableDPFKey.empty_like(key.dpf_key))
        key.c = torch.empty_like(key.ver_dpf_key.s, dtype=torch.bool, device=DEVICE)
        key.size = key.c.shape[0]
        return key


# verfiable MSB protocol from sigma protocol without r_out
def verifiable_sigma_gen(num_of_keys):
    r_in = RingTensor.random([num_of_keys])
    x1 = r_in
    y1 = r_in % (HALF_RING - 1)
    k0, k1 = VerifiableDPF.gen(num_of_keys, y1, RingTensor.convert_to_ring(1))
    c = x1.signbit() ^ 1
    c0 = torch.randint(0, 1, [num_of_keys], device=DEVICE)
    c0 = RingTensor.convert_to_ring(c0)
    c1 = c ^ c0

    k0 = VerSigmaKey(k0)
    k1 = VerSigmaKey(k1)

    k0.c = c0
    k1.c = c1

    k0.r_in, k1.r_in = ArithmeticSharedRingTensor.share(r_in, 2)

    return k0, k1


def verifiable_sigma_eval(party_id, key, x_shift: RingTensor):
    shape = x_shift.shape
    x_shift = x_shift.reshape(-1, 1)
    K, c = key
    y = x_shift % (HALF_RING - 1)
    y = y + 1
    out, pi = VerifiableDPF.ppq(y, K, party_id)
    out = x_shift.signbit() * party_id ^ c.reshape(-1, 1) ^ out
    return out.reshape(shape), pi
