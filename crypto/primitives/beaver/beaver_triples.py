import operator
import os
import random
import re
from functools import reduce

import torch

from common.utils.cuda_utils import cuda_matmul
from config.base_configs import DEBUG, data_type, DEVICE, DTYPE, SCALE, base_path
from crypto.tensor.ArithmeticSharedRingTensor import ArithmeticSharedRingTensor
from crypto.primitives.arithmetic_secret_sharing.replicated_secret_sharing import ReplicatedSecretSharing
from crypto.primitives.auxiliary_parameter.param_file_pipe import ParamFilePipe
from crypto.primitives.auxiliary_parameter.parameter import Parameter
from crypto.primitives.homomorphic_encryption.Paillier import Paillier
from crypto.primitives.auxiliary_parameter.param_buffer import ParamBuffer
from crypto.tensor.RingTensor import RingTensor


class BeaverTriples(Parameter):
    def __init__(self):
        self.a = None
        self.b = None
        self.c = None
        self.size = 0
        self.share_type = "22"

    def __iter__(self):
        return iter((self.a, self.b, self.c))

    def set_triples(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    @staticmethod
    def gen(*args):
        """
        args[0]: type of generation \n
        args[1]: number of party \n
        args[2]: number of parameters \n
        args[3]: x_shape if MAT or party if HE \n
        args[4]: y_shape if MAT \n
        :return:
        """
        if args[1] == 2:
            if args[0] == 'HE':
                return gen_triples_by_homomorphic_encryption(args[2], args[3])
            elif args[0] == 'TTP':
                return gen_triples_by_ttp(args[2])
            elif args[0] == 'MAT':
                return gen_matrix_triples_by_ttp(args[2], args[3], args[4])

        elif args[1] == 3:
            return gen_triples_malicious(args[2])

    @classmethod
    def gen_and_save(cls, *args):
        """
        args[0]: type of generation \n
        args[1]: number of party \n
        args[2]: number of parameters \n
        args[3]: x_shape if MAT or party if HE \n
        args[4]: y_shape if MAT \n
        :return:
        """
        triples = cls.gen(*args)
        if args[0] == 'TTP' or args[0] == 'MTP':
            for party_id in range(args[1]):
                ParamFilePipe.write(triples[party_id], party_id, args[1])
        elif args[0] == 'HE':
            file_path = base_path + f"{args[1]}party/aux_parameters/BeaverTriples"

            file_names = os.listdir(file_path)
            max_ptr = 0
            for fname in file_names:
                match = re.search(r"BeaverTriples_\d+_(\d+)\.pth", fname)
                if match:
                    max_ptr = max(max_ptr, int(match.group(1)))

            file_name = f"BeaverTriples_{args[3].party_id}_{max_ptr + 1}.pth"
            ParamFilePipe.write_by_name(triples, file_name, file_path)
        elif args[0] == 'MAT':
            for party_id in range(args[1]):
                file_path = base_path + f"/aux_parameters/BeaverTriples/{args[1]}party/Matrix"
                file_name = f"MatrixBeaverTriples_{party_id}_{list(args[3])}_{list(args[4])}.pth"
                ParamFilePipe.write_by_name(triples[party_id], file_name, file_path)


def gen_triples_by_ttp(num_of_triples):
    """
    可信第三方生成乘法Beaver三元组
    :param num_of_triples: 三元组个数
    :return: 乘法Beaver三元组
    """

    a = RingTensor.random([num_of_triples])
    b = RingTensor.random([num_of_triples])
    c = a * b

    a_list = ArithmeticSharedRingTensor.share(a, 2)
    b_list = ArithmeticSharedRingTensor.share(b, 2)
    c_list = ArithmeticSharedRingTensor.share(c, 2)

    triples = []
    for i in range(2):
        triples.append(BeaverTriples())
        triples[i].share_type = "22"
        triples[i].a = ArithmeticSharedRingTensor(a_list[i].to('cpu'), None)
        triples[i].b = ArithmeticSharedRingTensor(b_list[i].to('cpu'), None)
        triples[i].c = ArithmeticSharedRingTensor(c_list[i].to('cpu'), None)
        triples[i].size = num_of_triples

    return triples


def gen_triples_malicious(num_of_triples):
    """
    恶意参与方生成乘法Beaver三元组
    :param num_of_triples: 三元组个数
    :return: 乘法Beaver三元组
    """
    # TODO: 现在还是半诚实，后续改恶意
    a = RingTensor.random([num_of_triples])
    b = RingTensor.random([num_of_triples])
    c = a * b

    a_list = ReplicatedSecretSharing.share(a)
    b_list = ReplicatedSecretSharing.share(b)
    c_list = ReplicatedSecretSharing.share(c)

    triples = []
    for i in range(3):
        triples.append(BeaverTriples())
        triples[i].share_type = "32"
        triples[i].a = a_list[i]
        triples[i].b = b_list[i]
        triples[i].c = c_list[i]
        triples[i].size = num_of_triples

    return triples


def gen_triples_by_homomorphic_encryption(num_of_triples, party):
    """
    同态加密生成乘法Beaver三元组
    :param num_of_triples: 三元组个数
    :param party: 参与方
    :return: 乘法Beaver三元组
    """
    # a = [random.randint(-HALF_RING, HALF_RING - 1) for _ in range(num_of_triples)]
    # b = [random.randint(-HALF_RING, HALF_RING - 1) for _ in range(num_of_triples)]
    # c = []
    a = [random.randint(0, 2 ^ 32) for _ in range(num_of_triples)]
    b = [random.randint(0, 2 ^ 32) for _ in range(num_of_triples)]
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
        c = [decrypted_d[i] + a[i] * b[i] for i in range(num_of_triples)]

    elif party.party_id == 1:
        # r = [random.randint(-HALF_RING, HALF_RING - 1) for _ in range(num_of_triples)]
        r = [random.randint(0, 2 ^ 32) for _ in range(num_of_triples)]
        c = [a[i] * b[i] - r[i] for i in range(num_of_triples)]

        messages = party.receive()

        encrypted_r = Paillier.encrypt_with_key(r, messages[2])
        d = [messages[0][i] ** b[i] * messages[1][i] ** a[i] * encrypted_r[i] for i in range(num_of_triples)]
        # send d to party 0
        party.send(d)

    triples = BeaverTriples()
    triples.a = RingTensor(torch.tensor(a)).to('cpu')
    triples.b = RingTensor(torch.tensor(b)).to('cpu')
    triples.c = RingTensor(torch.tensor(c)).to('cpu')
    triples.size = num_of_triples

    return triples


def gen_matrix_triples_by_ttp(num_of_param, x_shape, y_shape, num_of_party=2):
    """
    生成乘法Beaver三元组
    :param num_of_param: 需要矩阵三元组的个数:
    :param x_shape: 矩阵x的形状
    :param y_shape: 矩阵y的形状
    :param num_of_party: 参与方个数
    :return: 乘法Beaver三元组
    """
    x_shape = [num_of_param] + list(x_shape)
    y_shape = [num_of_param] + list(y_shape)
    a = RingTensor.random(x_shape)
    b = RingTensor.random(y_shape)
    if a.device == 'cpu':
        c = a @ b
    else:
        c = cuda_matmul(a.tensor, b.tensor)
        c = RingTensor.convert_to_ring(c)

    a_list = ArithmeticSharedRingTensor.share(a, num_of_party)
    b_list = ArithmeticSharedRingTensor.share(b, num_of_party)
    c_list = ArithmeticSharedRingTensor.share(c, num_of_party)

    triples = []
    for i in range(num_of_party):
        triples.append(BeaverTriples())
        triples[i].a = ArithmeticSharedRingTensor(a_list[i].to('cpu'), None)
        triples[i].b = ArithmeticSharedRingTensor(b_list[i].to('cpu'), None)
        triples[i].c = ArithmeticSharedRingTensor(c_list[i].to('cpu'), None)

    return triples


class BeaverBuffer(ParamBuffer):
    def __init__(self, party=None):
        super().__init__(party=party)
        self.party = party
        self.param_type = BeaverTriples
        self.mat_beaver = {}
        self.matrix_ptr = 0
        self.matrix_ptr_max = 0

    def get_triples(self, shape):
        # 这里先做一个统一接口，所有Provider都使用get_triples接口获取任意shape的三元组，包括各种类型
        # 需要的三元组数量
        number_of_triples = reduce(operator.mul, shape, 1)
        a, b, c = self.get_parameters(number_of_triples)
        a.party = self.party
        b.party = self.party
        c.party = self.party
        if DEBUG:
            return a, b, c
        return a.reshape(shape), b.reshape(shape), c.reshape(shape)

    def load_mat_beaver(self):
        self.matrix_ptr_max = len(self.mat_beaver)

    def load_mat_beaver_from_file(self, x_shape, y_shape, num_of_party=2):
        file_name = f"MatrixBeaverTriples_{self.party.party_id}_{list(x_shape)}_{list(y_shape)}.pth"
        file_path = base_path + f"/aux_parameters/BeaverTriples/{num_of_party}party/Matrix"
        try:
            mat_triples_dic = ParamFilePipe.read_dic_by_name(file_name, file_path)
            self.mat_beaver = BeaverTriples.from_dic(mat_triples_dic)
            self.mat_beaver.a.party = self.party
            self.mat_beaver.b.party = self.party
            self.mat_beaver.c.party = self.party
        except FileNotFoundError:
            raise Exception("Need generate matrix triples in this shape first!")

    def get_mat_beaver(self, x_shape, y_shape):
        if DEBUG:
            if f"{list(x_shape)}_{list(y_shape)}" not in self.mat_beaver.keys():
                a = torch.ones(x_shape, dtype=data_type, device=DEVICE)
                b = torch.ones(y_shape, dtype=data_type, device=DEVICE)

                assert x_shape[-1] == y_shape[-2]
                c_shape = torch.broadcast_shapes(x_shape[:-2], y_shape[:-2]) + (x_shape[-2], y_shape[-1])
                c = torch.ones(c_shape, dtype=data_type, device=DEVICE) * x_shape[-1]

                a_tensor = RingTensor(a, dtype=DTYPE).to(DEVICE)
                b_tensor = RingTensor(b, dtype=DTYPE).to(DEVICE)
                c_tensor = RingTensor(c, dtype=DTYPE).to(DEVICE)

                a = ArithmeticSharedRingTensor(a_tensor, self.party)
                b = ArithmeticSharedRingTensor(b_tensor, self.party)
                c = ArithmeticSharedRingTensor(c_tensor, self.party)

                self.mat_beaver[f"{list(x_shape)}_{list(y_shape)}"] = BeaverTriples()
                self.mat_beaver[f"{list(x_shape)}_{list(y_shape)}"].set_triples(a, b, c)
            return self.mat_beaver[f"{list(x_shape)}_{list(y_shape)}"]
        else:
            mat_beaver = self.mat_beaver[self.matrix_ptr].pop()
            self.matrix_ptr = (self.matrix_ptr + 1) % self.matrix_ptr_max
            return mat_beaver
