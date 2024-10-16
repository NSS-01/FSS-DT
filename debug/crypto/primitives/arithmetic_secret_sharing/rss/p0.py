import time

from NssMPC import RingTensor
from NssMPC.config.configs import DEVICE
from NssMPC.crypto.primitives.arithmetic_secret_sharing.replicated_secret_sharing import ReplicatedSecretSharing
from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional import v_matmul_with_trunc
from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional.compare import secure_ge
from NssMPC.crypto.protocols.replicated_secret_sharing.semi_honest_functional import matmul_with_trunc
from NssMPC.secure_model.mpc_party import SemiHonest3PCParty, HonestMajorityParty

import torch

id = 0

Party = HonestMajorityParty(id=id)
# Party = SemiHonest3PCParty(id=id)
Party.set_comparison_provider()
Party.set_multiplication_provider()
Party.set_trunc_provider()
Party.online()

share_table = ReplicatedSecretSharing([RingTensor([-1, -1]), RingTensor([2, 3])], Party)
share_table.restore()
# 创建一个Ringtensor进行测试
x = torch.tensor([[1.1, 1.1, 1.3]], device=DEVICE)
y = torch.tensor([[1.2], [1.1], [2.3]], device=DEVICE)
print(x)
print(y)
print( x @ y)
x = RingTensor.convert_to_ring(x)
y = RingTensor.convert_to_ring(y)


# 转换为RSS
shares_X = ReplicatedSecretSharing.share(x)
shares_Y = ReplicatedSecretSharing.share(y)

# p0将v1 v2 发送给p1
Party.send(1, shares_X[1])
Party.send(2, shares_X[2])

# p0将v1 v2 发送给p1
Party.send(1, shares_Y[1])
Party.send(2, shares_Y[2])

shared_x = shares_X[0]
shared_y = shares_Y[0]
shared_x.party = Party
shared_y.party = Party
print(shared_x.restore())

print("shared_x: ", shared_x.shape)
print("shared_y: ", shared_y.shape)



res = v_matmul_with_trunc(shared_x, shared_y)

print("res: ", res.restore().convert_to_real_field())

res_ori = shared_x @ shared_y
print("res_ori: ", res.restore().convert_to_real_field())


# add_result = shared_tensor + shared_tensor
#
# res = add_result.restore()
# print(res.convert_to_real_field())

# print(x > y)
# res0 = x > y
#
# # mul_result = shared_x > shared_y
# res = secure_ge(shared_x, shared_y)
# # mul_result = mul_result.view(-1, 1)
# res = res.restore()
# res1 = res.convert_to_real_field()
# print(res)
# print("res", res1)

# delta = res0.to("cuda") - res1
# error = torch.where(delta.abs() > 1, torch.tensor(1, device='cuda'), torch.tensor(0, device='cuda'))
# print("error", error.sum())
