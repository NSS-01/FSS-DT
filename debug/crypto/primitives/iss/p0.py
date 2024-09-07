import torch

from crypto.mpc.semi_honest_party import *
from config.mpc_configs import *

from crypto.tensor.RingTensor import RingTensor
from crypto.primitives.arithmetic_secret_sharing_dev import *

import time

id = 0
mapping = {1: (S1_ADDRESS, S1_client_port_to_S0), 2: (S2_ADDRESS, S2_client_port_to_S0)}

# 创建一个参与方Party，它的编号是0，它的总参与方数是3
Party = SemiHonestMPCParty(id = id, parties_num = 3)
Party.set_scale(SCALE)
Party.set_dtype(DTYPE)
Party.start_server(S0_ADDRESS, S0_PORT)
Party.set_target_client_mapping(mapping)

mask_path = './data/mask_data/S0/'
Party.load_pre_generated_mask(mask_path)
# print(Party.pre_generated_mask.restore().convert_to_real_field())
# 参与方Party0与其他两方进行链接
Party.connect_to_other(1, S0_ADDRESS, S0_client_port_to_S1, S1_ADDRESS, S1_PORT)
Party.connect_to_other(2, S0_ADDRESS, S0_client_port_to_S2, S2_ADDRESS, S2_PORT)

Party.check_all_parties_online()


# 创建一个Ringtensor进行测试
test_value = torch.randint(0, 10, [2 , 2])
# test_value = torch.tensor([[0,2,3,4],[5,6,-7,-8]], dtype=torch.int32)
test_tensor = RingTensor.convert_to_ring(test_value)
print("test_tensor: ", test_tensor)


# 创建一个Ringtensor进行测试
test_value2 = torch.randint(0, 10, [2 , 2])
# test_value2 = torch.tensor([[0,1,2,5],[6,12,-12,-5]], dtype=torch.int32)
test_tensor2 = RingTensor.convert_to_ring(test_value2)
print("test_tensor2: ", test_tensor)

# 转换为ISS
masked_tensor, shares = ImprovedSecretSharing.share(test_tensor)
print(masked_tensor + shares[0][0] + shares[1][0] + shares[2][0])
shared_tensor0 = ReplicatedSecretSharing(replicated_shared_tensor = shares[0], party=Party)
shared_tensor1 = ReplicatedSecretSharing(replicated_shared_tensor = shares[1], party=Party)
shared_tensor2 = ReplicatedSecretSharing(replicated_shared_tensor = shares[2], party=Party)

iss_0 = ImprovedSecretSharing(masked_tensor, shared_tensor0, Party)
iss_1 = ImprovedSecretSharing(masked_tensor, shared_tensor1, Party)
iss_2 = ImprovedSecretSharing(masked_tensor, shared_tensor2, Party)

# p0将v1 v2 发送给p1
Party.send_iss_to(1, iss_1)
Party.send_iss_to(2, iss_2)

shared_tensor = iss_0
print(shared_tensor.restore().convert_to_real_field())



# 转换为ISS
masked_tensor, shares = ImprovedSecretSharing.share(test_tensor2)
print(masked_tensor + shares[0][0] + shares[1][0] + shares[2][0])
shared_tensor0 = ReplicatedSecretSharing(replicated_shared_tensor = shares[0], party=Party)
shared_tensor1 = ReplicatedSecretSharing(replicated_shared_tensor = shares[1], party=Party)
shared_tensor2 = ReplicatedSecretSharing(replicated_shared_tensor = shares[2], party=Party)

iss_0 = ImprovedSecretSharing(masked_tensor, shared_tensor0, Party)
iss_1 = ImprovedSecretSharing(masked_tensor, shared_tensor1, Party)
iss_2 = ImprovedSecretSharing(masked_tensor, shared_tensor2, Party)

# p0将v1 v2 发送给p1
Party.send_iss_to(1, iss_1)
Party.send_iss_to(2, iss_2)

shared_tensor2 = iss_0
print(shared_tensor2.restore().convert_to_real_field())

a_sl = shared_tensor[1]
print(a_sl.restore().convert_to_real_field())
#
# add_result = shared_tensor + shared_tensor
#
# res = add_result.restore()
# print(res.convert_to_real_field())
#
#
mul_result = shared_tensor * 2
res = mul_result.restore()
print(res.convert_to_real_field())

mul_result2 = mul_result * shared_tensor2
res = mul_result2.restore()
print(res.convert_to_real_field())
#
# sum_result = shared_tensor.sum(0)
# res = sum_result.restore()
# print(res.convert_to_real_field())

# o = iss_DICF(shared_tensor)
# print(o.restore().convert_to_real_field())
# ge_result = shared_tensor >= shared_tensor2
# res = ge_result.restore()
# print((test_value >= test_value2) + 0)
# print(res.convert_to_real_field())
#
# lt_result = shared_tensor <= shared_tensor2
# res = lt_result.restore()
# print((test_value <= test_value2) + 0)
# print(res.convert_to_real_field())
#


ID_tensor = torch.ones([2,2], dtype=torch.int64) * -1
ID_Ring = RingTensor.convert_to_ring(ID_tensor)
ID_Public, ID_shares = ImprovedSecretSharing.share(ID_Ring)
print(ID_Public + ID_shares[0][0] + ID_shares[1][0] + ID_shares[2][0])
shared_tensor0 = ReplicatedSecretSharing(replicated_shared_tensor = ID_shares[0], party=Party)
shared_tensor1 = ReplicatedSecretSharing(replicated_shared_tensor = ID_shares[1], party=Party)
shared_tensor2 = ReplicatedSecretSharing(replicated_shared_tensor = ID_shares[2], party=Party)

iss_0 = ImprovedSecretSharing(ID_Public, shared_tensor0, Party)
iss_1 = ImprovedSecretSharing(ID_Public, shared_tensor1, Party)
iss_2 = ImprovedSecretSharing(ID_Public, shared_tensor2, Party)

# p0将v1 v2 发送给p1
Party.send_iss_to(1, iss_1)
Party.send_iss_to(2, iss_2)

ID = iss_0
print(ID.restore().convert_to_real_field())

u = shared_tensor[0]
print(u.restore().convert_to_real_field())

ID[:,1] = u
print(ID.restore().convert_to_real_field())