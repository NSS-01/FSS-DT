import time
import unittest

from NssMPC.config import DEBUG_LEVEL
from NssMPC.crypto.aux_parameter import RssMatmulTriples
from NssMPC.crypto.primitives.arithmetic_secret_sharing.replicated_secret_sharing import ReplicatedSecretSharing
from NssMPC.crypto.protocols.replicated_secret_sharing.semi_honest_functional import *
from NssMPC.secure_model.mpc_party.semi_honest import SemiHonest3PCParty

# 定义测试规模
num = 10
print("测试数据规模: ", num, "*", num)

id = 2
Party = SemiHonest3PCParty(id=id)
Party.set_comparison_provider()
Party.set_trunc_provider()
Party.online()
x = ReplicatedSecretSharing.random([1000, 256], party=Party)
y = ReplicatedSecretSharing.random([256, 100], party=Party)
triples = RssMatmulTriples.gen(1, x.shape, y.shape)
if DEBUG_LEVEL != 2:  # TODO: DEBUG_LEVEL统一
    triple = Party.receive(0)
    Party.providers[RssMatmulTriples.__name__].param = [triple]
    Party.providers[RssMatmulTriples.__name__].load_mat_beaver()

class TestRss(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        Party.close()

    # def test_01_get_item(self):
    #     start_time = time.time()
    #     _ = x[num // 2]
    #     end_time = time.time()
    #     print("get item time: ", end_time - start_time)
    #
    # def test_02_set_item(self):
    #     r = ReplicatedSecretSharing.random([1], party=Party)
    #     start_time = time.time()
    #     x[num // 2][num // 2] = r
    #     end_time = time.time()
    #     print("set item time: ", end_time - start_time)
    #
    # def test_03_add(self):
    #     start_time = time.time()
    #     _ = x + y
    #     end_time = time.time()
    #     print("add time: ", end_time - start_time)
    #
    # def test_04_mul(self):
    #     start_time = time.time()
    #     _ = x * y
    #     end_time = time.time()
    #     print("mul time: ", end_time - start_time)


    def test_04_mat_mul(self):
        for i in range(10):
            start_time = time.time()
            _ = x @ y
            end_time = time.time()
            print("mul time: ", end_time - start_time)

    def test_04_mat_mul_with(self):
        for i in range(10):
            start_time = time.time()
            _ = matmul_with_trunc(x, y)
            end_time = time.time()
            print("mul time: ", end_time - start_time)

    # def test_05_ge(self):
    #     start_time = time.time()
    #     _ = x >= y
    #     end_time = time.time()
    #     print("ge time: ", end_time - start_time)
    #
    # def test_06_rand(self):
    #     start_time = time.time()
    #     _ = rand([num, num], party=Party)
    #     end_time = time.time()
    #     print("ge rand: ", end_time - start_time)
    #
    # def test_07_b2a(self):
    #     brss = ReplicatedSecretSharing.zeros([num, num], party=Party)
    #     start_time = time.time()
    #     _ = bit_injection(brss)
    #     end_time = time.time()
    #     print("bit injection time: ", end_time - start_time)
    #
    # def test_08_restore(self):
    #     start_time = time.time()
    #     _ = x.restore()
    #     end_time = time.time()
    #     print("restore time: ", end_time - start_time)
