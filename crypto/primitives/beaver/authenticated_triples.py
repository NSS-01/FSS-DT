"""
参考 VerifyML: Obliviously Checking Model Fairness Resilient to Malicious Model Holder
https://ieeexplore.ieee.org/abstract/document/10168299
"""

import random

import torch

from crypto.tensor.ArithmeticSharedRingTensor import ArithmeticSharedRingTensor
from crypto.primitives.arithmetic_secret_sharing.replicated_secret_sharing import ReplicatedSecretSharing
from crypto.primitives.homomorphic_encryption.Paillier import Paillier
from crypto.tensor.RingTensor import RingTensor, cuda_matmul


class AuthenticatedTriples(object):
    def __init__(self):
        self.a = None
        self.b = None
        self.c = None
        self.mac_key = None
        self.mac_a = None
        self.mac_b = None
        self.mac_c = None
        self.size = 0

    def __getitem__(self, item):
        triples = AuthenticatedTriples()
        triples.a = self.a[item]
        triples.b = self.b[item]
        triples.c = self.c[item]
        triples.mac_key = self.mac_key[item]
        triples.mac_a = self.mac_a[item]
        triples.mac_b = self.mac_b[item]
        triples.mac_c = self.mac_c[item]
        return triples

    def __setitem__(self, key, value):
        self.a[key] = value.a
        self.b[key] = value.b
        self.c[key] = value.c
        self.mac_key[key] = value.mac_key
        self.mac_a[key] = value.mac_a
        self.mac_b[key] = value.mac_b
        self.mac_c[key] = value.mac_c

    def to(self, device):
        self.a = self.a.to(device)
        self.b = self.b.to(device)
        self.c = self.c.to(device)
        self.mac_key = self.mac_key.to(device)
        self.mac_a = self.mac_a.to(device)
        self.mac_b = self.mac_b.to(device)
        self.mac_c = self.mac_c.to(device)
        return self

    def to_dic(self):
        dict = {}
        dict['a'] = self.a
        dict['b'] = self.b
        dict['c'] = self.c
        dict['mac_key'] = self.mac_key
        dict['mac_a'] = self.mac_a
        dict['mac_b'] = self.mac_b
        dict['mac_c'] = self.mac_c
        return dict

    @staticmethod
    def dic_to_triples(dic):
        triples = AuthenticatedTriples()
        triples.a = dic['a']
        triples.b = dic['b']
        triples.c = dic['c']
        triples.mac_key = dic['mac_key']
        triples.mac_a = dic['mac_a']
        triples.mac_b = dic['mac_b']
        triples.mac_c = dic['mac_c']
        triples.size = triples.a.shape[0]
        return triples

    @staticmethod
    def empty(size):
        triples = AuthenticatedTriples()
        triples.a = RingTensor.empty(size)
        triples.b = RingTensor.empty(size)
        triples.c = RingTensor.empty(size)
        triples.mac_key = RingTensor.empty(size)
        triples.mac_a = RingTensor.empty(size)
        triples.mac_b = RingTensor.empty(size)
        triples.mac_c = RingTensor.empty(size)
        triples.size = size
        return triples

    @staticmethod
    def empty_like(other_triple):
        triples = AuthenticatedTriples()
        triples.a = RingTensor.empty_like(other_triple.a)
        triples.b = RingTensor.empty_like(other_triple.b)
        triples.c = RingTensor.empty_like(other_triple.c)
        triples.mac_key = RingTensor.empty_like(other_triple.mac_key)
        triples.mac_a = RingTensor.empty_like(other_triple.mac_a)
        triples.mac_b = RingTensor.empty_like(other_triple.mac_b)
        triples.mac_c = RingTensor.empty_like(other_triple.mac_c)
        triples.size = other_triple.size
        return triples


def gen_triples(num_of_triples, num_of_party=2):
    """
    可信第三方生成乘法Beaver三元组 TODO: 先这么写，后续改成各参与方共同生成
    :param num_of_triples: 三元组个数
    :param num_of_party: 参与方个数
    :return: 乘法Beaver三元组
    """

    a = RingTensor.random([num_of_triples])
    b = RingTensor.random([num_of_triples])
    c = a * b

    mac_key = RingTensor.random([num_of_triples])
    mac_a = mac_key * a
    mac_b = mac_key * b
    mac_c = mac_key * c

    a_list = ArithmeticSharedRingTensor.share(a, num_of_party)
    b_list = ArithmeticSharedRingTensor.share(b, num_of_party)
    c_list = ArithmeticSharedRingTensor.share(c, num_of_party)
    mac_key_list = ArithmeticSharedRingTensor.share(mac_key, num_of_party)
    mac_a_list = ArithmeticSharedRingTensor.share(mac_a, num_of_party)
    mac_b_list = ArithmeticSharedRingTensor.share(mac_b, num_of_party)
    mac_c_list = ArithmeticSharedRingTensor.share(mac_c, num_of_party)

    triples = []
    for i in range(num_of_party):
        triples.append(AuthenticatedTriples())
        triples[i].a = a_list[i].to('cpu')
        triples[i].b = b_list[i].to('cpu')
        triples[i].c = c_list[i].to('cpu')
        triples[i].mac_key = mac_key_list[i].to('cpu')
        triples[i].mac_a = mac_a_list[i].to('cpu')
        triples[i].mac_b = mac_b_list[i].to('cpu')
        triples[i].mac_c = mac_c_list[i].to('cpu')
        triples[i].size = num_of_triples

    return triples


def gen_matrix_triples(x_shape, y_shape, num_of_party=2):
    pass
