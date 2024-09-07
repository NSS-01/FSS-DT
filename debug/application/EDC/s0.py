from application.clustering.PLE_DBSCAN.protocols.edc_parallelization import *
from config.mpc_configs import *
from crypto.mpc.semi_honest_party import *

party_id = 0
# dataset_name = 'aggregation'

mapping = {1: (S1_ADDRESS, S1_client_port_to_S0), 2: (S2_ADDRESS, S2_client_port_to_S0)}

# 创建一个参与方Party，它的编号是0，它的总参与方数是3
Party = SemiHonestMPCParty(id=party_id, parties_num=3)
Party.set_parity_dicf_key_provider()
Party.set_scale(SCALE)
Party.set_dtype(DTYPE)
Party.start_server(S0_ADDRESS, S0_PORT)
Party.set_target_client_mapping(mapping)
Party.set_rss_provider()
# Party.rss_provider.gen_random_rss(10000)
Party.load_rss_provider(10000)

Party.connect_to_other(1, S0_ADDRESS, S0_client_port_to_S1, S1_ADDRESS, S1_PORT)
Party.connect_to_other(2, S0_ADDRESS, S0_client_port_to_S2, S2_ADDRESS, S2_PORT)

Party.check_all_parties_online()

# 定义点的维数N,这个量大伙都是知道的
N = 2

# 定义并行化的数量num_of_keys
num_of_points = 10

# offline测试开始
(k_0_PHIb, c_bAdd1_b), (k_1_PHIb, c_bAdd2_b), z, z_t = EDC_offline(Party, N, num_of_points)

# print("offline result c = ", c_bAdd1_b)
# print("offline result key = ", k_0_PHIb)

Party.send_ring_tensor_to((Party.party_id + 1) % 3, c_bAdd1_b)
Party.send_ring_tensor_to((Party.party_id + 2) % 3, c_bAdd2_b)

c_b_b1 = Party.receive_ring_tensor_from((Party.party_id + 1) % 3)
c_b_b2 = Party.receive_ring_tensor_from((Party.party_id + 2) % 3)

# print("交互完成")
# Offline测试完成
if i == 11:
    print("第11轮结束后")
    print(pla_S)
    print(pla_s)
    print(pla_eta)
    print(pla_pc)
    print(pla_connect)
    print(pla_IDu)
    print(pla_current_is_core)
    print(pla_ID)
    print(pla_S_row)
    print(pla_S_col)
if i == 10:
    print("第11轮开始前")
    print(pla_ID)
    print(pla_S)

# Online测试开始
p1 = torch.tensor([[400, 4],
                   [4, 4],
                   [4, 4],
                   [4, 4],
                   [4, 4],
                   [4, 4],
                   [4, 4],
                   [4, 4],
                   [4, 4],
                   [4, 4]])
p2 = torch.tensor([[8, 6],
                   [8, 6],
                   [4, 4],
                   [4, 4],
                   [4, 4],
                   [4, 4],
                   [4, 4],
                   [4, 4],
                   [4, 4],
                   [8, 6]])

p1 = RingTensor.convert_to_ring(p1)
p2 = RingTensor.convert_to_ring(p2)

public_p1, psi_p1 = ImprovedSecretSharing.share(p1)
psi_p10, psi_p11, psi_p12 = psi_p1
psi_p10 = ReplicatedSecretSharing(psi_p10, Party)
p1 = ImprovedSecretSharing(public_p1, psi_p10, Party)

p11 = ImprovedSecretSharing(public_p1, ReplicatedSecretSharing(psi_p11, Party), Party)
p12 = ImprovedSecretSharing(public_p1, ReplicatedSecretSharing(psi_p12, Party), Party)

Party.send_iss_to(1, p11)
Party.send_iss_to(2, p12)

public_p2, psi_p2 = ImprovedSecretSharing.share(p2)
psi_p20, psi_p21, psi_p22 = psi_p2
psi_p20 = ReplicatedSecretSharing(psi_p20, Party)
p2 = ImprovedSecretSharing(public_p2, psi_p20, Party)

p21 = ImprovedSecretSharing(public_p2, ReplicatedSecretSharing(psi_p21, Party), Party)
p22 = ImprovedSecretSharing(public_p2, ReplicatedSecretSharing(psi_p22, Party), Party)

Party.send_iss_to(1, p21)
Party.send_iss_to(2, p22)

minPts = RingTensor(torch.tensor([[10, 10, 10, 10, 10, 10, 10, 10, 10, 10]]))

public_minPts, psi_minPts = ImprovedSecretSharing.share(minPts)
psi_minPts0, psi_minPts1, psi_minPts2 = psi_minPts
psi_minPts0 = ReplicatedSecretSharing(psi_minPts0, Party)
minPts = ImprovedSecretSharing(public_minPts, psi_minPts0, Party)

minPts1 = ImprovedSecretSharing(public_minPts, ReplicatedSecretSharing(psi_minPts1, Party), Party)
minPts2 = ImprovedSecretSharing(public_minPts, ReplicatedSecretSharing(psi_minPts2, Party), Party)
Party.send_iss_to(1, minPts1)
Party.send_iss_to(2, minPts2)


c = EDC_online(minPts, p1, p2, N, z, z_t, c_b_b1, c_b_b2, Party)
# print(c)
c = RingTensor.convert_to_ring(c)

# # send c to others
Party.send_ring_tensor_to((Party.party_id + 1) % 3, c)
Party.send_ring_tensor_to((Party.party_id + 2) % 3, c)

c_1 = Party.receive_ring_tensor_from((Party.party_id - 1) % 3)
c_2 = Party.receive_ring_tensor_from((Party.party_id - 2) % 3)

# 向量化的时候
result = c.tensor ^ c_1.tensor ^ c_2.tensor
#
# # if result == 1:
# #     print("当前种子", torch.initial_seed())
#
print("result:", result)
#
# # online测试结束
#
# # # 三方OT测试
# # m0 = torch.tensor([0, 1, 0, 1, 0, 1, 1])
# # # print("m0:", m0)
# # m1 = torch.tensor([1, 0, 0, 0, 1, 0, 1])
# # # print("m1:", m1)
# # Parity3_OT_sender(m0, m1, Party)
# #三方OT测试结束
#
# # B2A测试
#
# c = RingTensor.convert_to_ring(torch.tensor([1, 1, 1, 0, 0, 0, 0]))
c = ReplicatedSecretSharing.reshare33(c, Party)

r0, r1, r2 = ReplicatedSecretSharing.share(RingTensor.convert_to_ring(torch.tensor([2, 1, 7, 1, 9, 3, 3, 3, 1, 10])))
r = ReplicatedSecretSharing(r0, Party)
r1 = ReplicatedSecretSharing(r1, Party)
r2 = ReplicatedSecretSharing(r2, Party)

Party.send_rss_to((Party.party_id + 1) % 3, r1)
Party.send_rss_to((Party.party_id + 2) % 3, r2)

result = B2A(c, r, Party)
print("B2A result:", result)

print("restore", result.restore())
