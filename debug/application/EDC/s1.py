import torch

from crypto.mpc.semi_honest_party import *
from config.mpc_configs import *
from application.clustering.PLE_DBSCAN.protocols.edc_parallelization import *
party_id = 1
# dataset_name = 'aggregation'

mapping = {0: (S0_ADDRESS, S0_client_port_to_S1), 2: (S2_ADDRESS, S2_client_port_to_S1)}

# 创建一个参与方Party，它的编号是1，它的总参与方数是3
Party = SemiHonestMPCParty(id=party_id, parties_num=3)
Party.set_parity_dicf_key_provider()
Party.set_scale(SCALE)
Party.set_dtype(DTYPE)
Party.start_server(S1_ADDRESS, S1_PORT)
Party.set_target_client_mapping(mapping)
Party.set_rss_provider()
Party.load_rss_provider(10000)

Party.connect_to_other(0, S1_ADDRESS, S1_client_port_to_S0, S0_ADDRESS, S0_PORT)
Party.connect_to_other(2, S1_ADDRESS, S1_client_port_to_S2, S2_ADDRESS, S2_PORT)

Party.check_all_parties_online()

# 定义点的维数N
N = 2

# 定义并行化的数量num_of_keys
num_of_points = 10

# offline测试开始
(k_0_PHIb, c_bAdd1_b), (k_1_PHIb, c_bAdd2_b), z, z_t = EDC_offline(Party, N, num_of_points)

print("offline result c = ", c_bAdd1_b)
print("offline result key = ", k_0_PHIb)

Party.send_ring_tensor_to((Party.party_id + 1) % 3, c_bAdd1_b)
Party.send_ring_tensor_to((Party.party_id + 2) % 3, c_bAdd2_b)

c_b_b1 = Party.receive_ring_tensor_from((Party.party_id + 1) % 3)
c_b_b2 = Party.receive_ring_tensor_from((Party.party_id + 2) % 3)

print("交互完成")
# offline测试结束

# online测试开始
p1 = Party.receive_iss_from(0)
p2 = Party.receive_iss_from(0)
minPts = Party.receive_iss_from(0)
# print(p1.restore())
# print(p2.restore())
# print(minPts.restore())
c = EDC_online(minPts, p1, p2, N, z, z_t, c_b_b1, c_b_b2, Party)
print(c)
c = RingTensor.convert_to_ring(c)

Party.send_ring_tensor_to((Party.party_id + 1) % 3, c)
Party.send_ring_tensor_to((Party.party_id + 2) % 3, c)

c_1 = Party.receive_ring_tensor_from((Party.party_id - 1) % 3)
c_2 = Party.receive_ring_tensor_from((Party.party_id - 2) % 3)

result = c.tensor ^ c_1.tensor ^ c_2.tensor

print("result:", result)
# # online测试结束
#
#
# # 3方OT测试
# # c = RingTensor.convert_to_ring(torch.tensor([1, 0, 0, 0, 1, 0, 1]))
# # mc = Parity3_OT_receiver(c, Party)
# # print("mc", mc)
# #3方OT测试结束
#
#
# #B2A测试
# c = RingTensor.convert_to_ring(torch.tensor([1, 1, 1, 0, 0, 0, 0]))
#
c = ReplicatedSecretSharing.reshare33(c,Party)
r = Party.receive_rss_from((Party.party_id - 1)%3)

result = B2A(c, r, Party)
print("B2A result:", result)

print("restore",result.restore())
