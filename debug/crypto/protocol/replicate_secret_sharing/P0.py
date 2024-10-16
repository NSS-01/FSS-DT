import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import NssMPC.application.neural_network as nn
from NssMPC.config import DEBUG_LEVEL
from NssMPC.secure_model.mpc_party.honest_majority import HonestMajorityParty
from NssMPC.secure_model.mpc_party.semi_honest import SemiHonest3PCParty

from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional import *

from NssMPC.crypto.protocols.replicated_secret_sharing.semi_honest_functional import *

Party = HonestMajorityParty(0)
Party.online()
# 测试恶意乘法
#
# A = torch.tensor([[1,2],[2,3]])
# B = torch.tensor([[3,4],[5,6]])
#
A = torch.tensor([[1.1, 2.2], [2.2, 3.3]])
B = torch.tensor([[3.3, 4.4], [5.5, 6.6]])
print(torch.matmul(A, B))
A = RingTensor.convert_to_ring(A)
print(A)
B = RingTensor.convert_to_ring(B)
print(B)

triples = RssMatmulTriples.gen(1, A.shape, B.shape)

if DEBUG_LEVEL != 2:    # TODO: DEBUG_LEVEL统一
    Party.send(1, triples[1])
    Party.send(2, triples[2])
    Party.providers[RssMatmulTriples.__name__].param = [triples[0]]
    Party.providers[RssMatmulTriples.__name__].load_mat_beaver()

# print(triples[0])
# print(triples[1])
# print(triples[2])

# print("*************************************")

X = share(A, Party)
print(X.restore().convert_to_real_field())
Y = share(B, Party)
print(Y.restore().convert_to_real_field())
#
mul = mul_with_out_trunc(X, Y)
mul = X * Y
print(mul.restore().convert_to_real_field())
mul = X @ Y
print(mul.restore().convert_to_real_field())

print("********************************")

#
# mul = v_mul(X, Y)
# print(mul.restore().convert_to_real_field())
#
# a = torch.randint(1,10,[2,2])
# b = torch.randint(1,10,[2,2])
# c = torch.matmul(a, b)
#
# a = RingTensor(a)
# b = RingTensor(b)
# c = RingTensor(c)
#
# aux_a = share(a, Party)
# aux_b = share(b, Party)
# aux_c = share(c, Party)
#
# print("aux a", aux_a.restore())
# print("aux b", aux_b.restore())
# print("aux c", aux_c.restore())
#
# # 生成随机用于验证的随机数
# mul = v_matmul(X,Y,(aux_a,aux_b,aux_c))
# print(mul.restore().convert_to_real_field())


# if DEBUG_LEVEL != 2:

#
# print('x @ y: ', x @ y)
# share_z = share_x @ share_y
# res_share_z = share_z.restore().convert_to_real_field()
# print('restored x @ y: ', res_share_z)
# assert torch.allclose(x @ y + .0, res_share_z, atol=1e-3, rtol=1e-3) == True
