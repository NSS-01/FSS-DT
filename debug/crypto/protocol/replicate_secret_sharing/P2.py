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

Party = HonestMajorityParty(2)
Party.online()
# 测试恶意乘法

if DEBUG_LEVEL != 2:    # TODO: DEBUG_LEVEL统一
    triple = Party.receive(0)
    # Party.append_provider()
    Party.providers[RssMatmulTriples.__name__].param = [triple]
    Party.providers[RssMatmulTriples.__name__].load_mat_beaver()

X = receive_share_from(0, Party)
print(X.restore().convert_to_real_field())
Y = receive_share_from(0, Party)
print(Y.restore().convert_to_real_field())
# mul = mul_with_out_trunc(X, Y)
# mul = X * Y
# mul = X @ Y
#

mul = X * Y
print(mul.restore().convert_to_real_field())
mul = X @ Y
print(mul.restore().convert_to_real_field())
print("********************************")
#
# mul = v_mul(X, Y)
# print(mul.restore().convert_to_real_field())
#
# aux_a = receive_share_from(0, Party)
# aux_b = receive_share_from(0, Party)
# aux_c = receive_share_from(0, Party)
#
# print("aux a", aux_a.restore())
# print("aux b", aux_b.restore())
# print("aux c", aux_c.restore())
#
# mul = v_matmul(X,Y, (aux_a,aux_b,aux_c))
# print(mul.restore().convert_to_real_field())
