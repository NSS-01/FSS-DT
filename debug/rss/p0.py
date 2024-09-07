import torch

from crypto.mpc.semi_honest_party import *
from config.mpc_configs import *
from config.base_configs import *

from crypto.tensor.RingTensor import RingTensor
from crypto.primitives.arithmetic_secret_sharing import *

import time

id = 0
mapping = {1: (S1_ADDRESS, S1_client_port_to_S0), 2: (S2_ADDRESS, S2_client_port_to_S0)}

# 创建一个参与方Party，它的编号是0，它的总参与方数是3
Party = SemiHonestMPCParty(id=id, parties_num=3)
Party.set_scale(SCALE)
Party.set_dtype(DTYPE)
Party.start_server(S0_ADDRESS, S0_PORT)
Party.set_target_client_mapping(mapping)

# 参与方Party0与其他两方进行链接
Party.connect_to_other(1, S0_ADDRESS, S0_client_port_to_S1, S1_ADDRESS, S1_PORT)
Party.connect_to_other(2, S0_ADDRESS, S0_client_port_to_S2, S2_ADDRESS, S2_PORT)

Party.check_all_parties_online()

# 创建一个Ringtensor进行测试
# test_value = torch.tensor([1,2,3,4,5,6,-7,-8,-9,-10], dtype=torch.int32)
# test_value = torch.randint(0, 100, [1000], dtype=torch.int64)
test_value = torch.rand([100000])
print(test_value)
test_tensor = RingTensor.convert_to_ring(test_value)
print("test_tensor: ", test_tensor)

# 转换为RSS
shares = ReplicatedSecretSharing.share(test_tensor)
# 此时shares里为[v0, v1, v2]
# p0持有[v0,v1]
shared_tensor0 = ReplicatedSecretSharing(replicated_shared_tensor=shares[0], party=Party)
shared_tensor1 = ReplicatedSecretSharing(replicated_shared_tensor=shares[1], party=Party)
shared_tensor2 = ReplicatedSecretSharing(replicated_shared_tensor=shares[2], party=Party)

# p0将v1 v2 发送给p1
Party.send_rss_to(1, shared_tensor1)
Party.send_rss_to(2, shared_tensor2)

shared_tensor = shared_tensor0

# add_result = shared_tensor + shared_tensor
#
# res = add_result.restore()
# print(res.convert_to_real_field())

print(test_value * test_value)
res0 = test_value * test_value
start = time.time()
mul_result = shared_tensor * shared_tensor
end = time.time()
res = mul_result.restore()
res1 = res.convert_to_real_field()
print("res", res1)

delta = res0.to("cuda") - res1
error = torch.where(delta.abs() > 1, torch.tensor(1, device='cuda'), torch.tensor(0, device='cuda'))
print("error", error.sum())
print(end - start)
