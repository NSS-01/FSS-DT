import os
import random
import re

import torch

from config.base_configs import base_path
from crypto.primitives.auxiliary_parameter.param_file_pipe import ParamFilePipe
from crypto.primitives.auxiliary_parameter.parameter import Parameter
from crypto.primitives.homomorphic_encryption.Paillier import Paillier


class MSBTriples(Parameter):
    def __init__(self, a=None, b=None, c=None):
        self.a = a
        self.b = b
        self.c = c
        self.size = 0

    def __iter__(self):
        return iter((self.a, self.b, self.c))

    @staticmethod
    def gen(*args):
        """
        args[0]: number of parameters \n
        args[1]: number of party \n
        args[2]: type of generation \n
        args[3]: party if HE \n
        :return:
        """
        if args[2] == 'HE':
            return gen_msb_triples_by_homomorphic_encryption(args[0], args[3])
        elif args[2] == 'TTP':
            return gen_msb_triples_by_ttp(args[0], args[1])

    @classmethod
    def gen_and_save(cls, *args):
        """
        args[0]: number of parameters \n
        args[1]: number of party \n
        args[2]: type of generation \n
        args[3]: party if HE \n
        :return:
        """
        assert args[2] in ['HE', 'TTP', 'TFP']
        triples = cls.gen(*args)
        if args[2] == 'HE':
            file_path = f"{base_path}/aux_parameters/MSBTriples/{args[1]}party/"

            file_names = os.listdir(file_path)
            max_ptr = 0
            for fname in file_names:
                match = re.search(r"MSBTriples_\d+_(\d+)\.pth", fname)
                if match:
                    max_ptr = max(max_ptr, int(match.group(1)))

            file_name = f"MSBTriples_{args[3].party_id}_{max_ptr + 1}.pth"
            ParamFilePipe.write_by_name(triples, file_name, file_path)

        elif args[2] == 'TTP':
            for party_id in range(args[1]):
                ParamFilePipe.write(triples[party_id], party_id, args[1])


def share(x, bit_len, num_of_party: int):
    """
    将二进制串进行秘密分享

    :param x: 要进行共享操作的张量
    :param bit_len: 二进制串长度
    :param num_of_party: 参与分享的参与方数量

    :return: 存放各秘密份额的列表(List)
    """

    share_x = []
    x_0 = x.clone()

    for i in range(num_of_party - 1):
        x_i = torch.randint(0, 2, [bit_len], dtype=torch.bool)
        share_x.append(x_i)
        x_0 ^= x_i
    share_x.append(x_0)
    return share_x


def gen_msb_triples_by_ttp(bit_len, num_of_party=2):
    """
    可信第三方生成msb三元组

    :param bit_len: 01串长度
    :param num_of_party: 参与方个数

    """
    a = torch.randint(0, 2, [bit_len], dtype=torch.bool)
    b = torch.randint(0, 2, [bit_len], dtype=torch.bool)
    c = a & b

    a_list = share(a, bit_len, num_of_party)
    b_list = share(b, bit_len, num_of_party)
    c_list = share(c, bit_len, num_of_party)

    triples = []
    for i in range(num_of_party):
        triples.append(MSBTriples())
        triples[i].a = a_list[i].to('cpu')
        triples[i].b = b_list[i].to('cpu')
        triples[i].c = c_list[i].to('cpu')
        triples[i].size = bit_len

    return triples


def gen_msb_triples_by_homomorphic_encryption(bit_len, party):
    """
    同态加密生成msb三元组

    :param bit_len: 01串长度
    :param party: 参与方
    """

    a = [random.randint(0, 2) for _ in range(bit_len)]
    b = [random.randint(0, 2) for _ in range(bit_len)]
    c = []

    if party.party_id == 0:
        paillier = Paillier()
        paillier.gen_keys()

        encrypted_a = paillier.encrypt(a)
        encrypted_b = paillier.encrypt(b)
        # send encrypted_a, encrypted_b to party 1
        party.send([encrypted_a, encrypted_b, paillier.public_key])
        # receive d from party 1
        d = party.receive()
        decrypted_d = paillier.decrypt(d)
        c = [decrypted_d[i] + a[i] * b[i] for i in range(bit_len)]

    elif party.party_id == 1:
        # r = [random.randint(-HALF_RING, HALF_RING - 1) for _ in range(num_of_triples)]
        r = [random.randint(0, 2) for _ in range(bit_len)]
        c = [a[i] * b[i] - r[i] for i in range(bit_len)]

        messages = party.receive()

        encrypted_r = Paillier.encrypt_with_key(r, messages[2])
        d = [messages[0][i] ** b[i] * messages[1][i] ** a[i] * encrypted_r[i] for i in range(bit_len)]
        # send d to party 0
        party.send(d)

    msb_triples = MSBTriples(torch.tensor(a, dtype=torch.bool).to('cpu'), torch.tensor(b, dtype=torch.bool).to('cpu'),
                             torch.tensor(c, dtype=torch.bool).to('cpu'))

    return msb_triples
