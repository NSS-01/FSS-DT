import operator
from functools import reduce

from config.base_configs import triple_path, DEBUG, DEVICE
from crypto.primitives.beaver.authenticated_triples import *
from crypto.primitives.auxiliary_parameter.param_buffer import ParamBuffer


class AuthenticatedBeaverBuffer(ParamBuffer):
    def __init__(self, party=None):
        super(AuthenticatedBeaverBuffer, self).__init__()
        self.party = party

    @staticmethod
    def gen_beaver_triples(num_of_triples, num_of_party=2):
        triples_lists = gen_triples(num_of_triples, num_of_party)
        for party_id in range(num_of_party):
            file_path = triple_path + f"{num_of_party}party/"
            file_name = f"auth_triples_{party_id}.pth"
            AuthenticatedBeaverBuffer.save_param_to_file(triples_lists[party_id], file_name, file_path)

    def load_triples(self, num_of_party=2):
        self.load_param(AuthenticatedTriples, f"{num_of_party}party/auth_triples_{self.party.party_id % 2}.pth",
                        triple_path)

    def load_beaver_buffers(self, num_of_party=2):
        self.load_buffers(AuthenticatedTriples,
                          f"{num_of_party}party/auth_triples_{self.party.party_id}_{self.file_ptr}.pth", triple_path)

    def get_triples_by_pointer(self, number_of_triples):
        if self.param is None:
            raise Exception("Please load triples first!")
        if DEBUG:
            triples = self.param[0].to(DEVICE)
            return triples.a, triples.b, triples.c, triples.mac_key, triples.mac_a, triples.mac_b, triples.mac_c
        if number_of_triples > self.param.size:
            raise Exception("Not enough triples!")
        end = self.ptr + number_of_triples
        triples = self.param[self.ptr: end].to(DEVICE)
        self.ptr = end
        return triples.a, triples.b, triples.c, triples.mac_key, triples.mac_a, triples.mac_b, triples.mac_c

    def get_triples(self, shape):
        # 这里先做一个统一接口，所有Provider都使用get_triples接口获取任意shape的三元组，包括各种类型
        # 需要的三元组数量
        number_of_triples = reduce(operator.mul, shape, 1)
        a, b, c, mac_key, mac_a, mac_b, mac_c = self.get_triples_by_pointer(number_of_triples)
        if DEBUG:
            return a, b, c, mac_key, mac_a, mac_b, mac_c
        return a.reshape(shape), b.reshape(shape), c.reshape(shape), mac_key.reshape(shape), mac_a.reshape(
            shape), mac_b.reshape(shape), mac_c.reshape(shape)
