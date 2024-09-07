from crypto.protocols.RSS_malicious_subprotocol.protocols import *
from crypto.protocols.CMP.verifiable_sigma import *
from crypto.primitives.function_secret_sharing.verifiable_dpf import VerifiableDPF, VerifiableDPFKey
from crypto.mpc.malicious_party import Malicious3PCParty
party_id = 0

num_of_elements = 1
# Pi 随机生成rdx
# rdx = RingTensor.random([num_of_elements])
rdx = RingTensor.convert_to_ring(torch.tensor([10], dtype=data_type))

# 使用rdx作为点函数的a点，生成vDPF密钥
K0, K1 = VerifiableDPF.gen(num_of_elements, alpha=rdx, beta=RingTensor.convert_to_ring(1))

a = RingTensor.convert_to_ring(torch.tensor([-100], dtype=data_type))

o1, _ = VerifiableDPF.ppq(a, K0.to(DEVICE), 0)
o2, _ = VerifiableDPF.ppq(a, K1.to(DEVICE), 1)

print(o1 ^ o2)

exit()


# 作为P0,初始化自身Party
Party = Malicious3PCParty(party_id = party_id)
Party.online()
Party.generate_prg_seed()

# 计算 x - y > 0 ?
# 首先使用恶意模型下的子协议，由P0生成x , y
x = RingTensor(torch.tensor([200], dtype=data_type))
y = RingTensor(torch.tensor([500], dtype=data_type))

X = share(x, Party)
Y = share(y, Party)


# 将rdx分享为(2,2)-secret sharing
rdx_shared_list = ArithmeticSecretSharing.share(rdx, num_of_party=2)
rdx0 = rdx_shared_list[0]
rdx1 = rdx_shared_list[1]

# 从本地加载其他地方发送的DPF
P1K1 = VerifiableDPFKey.load("P" + str((Party.party_id + 1) % 3) + "K1", './debug/application/malicious/FSSkeys')
P2K0 = VerifiableDPFKey.load("P" + str((Party.party_id + 2) % 3) + "K0", './debug/application/malicious/FSSkeys')

# 将rdx1， rdx0发送给P(i+1)， P(i+2)
Party.send_ring_tensor_to((Party.party_id + 1) % 3, rdx1)
Party.send_ring_tensor_to((Party.party_id + 2) % 3, rdx0)

# 从P(i+2)， P(i+1)接收rdx0， rdx1
rfromp2 = Party.receive_ring_tensor_from((Party.party_id + 2) % 3)
rfromp1 = Party.receive_ring_tensor_from((Party.party_id + 1) % 3)


# 计算X-Y
D = X - Y


# 设置rdx的秘密分享值
r0 = ReplicatedSecretSharing([rdx0, rdx1], Party)
r1 = ReplicatedSecretSharing([RingTensor.convert_to_ring(0), rfromp1], Party)
r2 = ReplicatedSecretSharing([rfromp2, RingTensor.convert_to_ring(0)], Party)

r0 = D + r0
r1 = D + r1
r2 = D + r2

rb1 = recon(r1, 0).to(DEVICE)
rb2 = recon(r2, 0).to(DEVICE)

recon(r2, 1)
recon(r0, 1)

recon(r0, 2)
recon(r1, 2)


beta1, pi1 = VerifiableDPF.ppq(rb1, P1K1.to(DEVICE), 1)
beta2, pi2 = VerifiableDPF.ppq(rb2, P2K0.to(DEVICE), 0)


Party.send_ring_tensor_to((Party.party_id + 2) % 3, pi1)
o_pi1 = Party.receive_ring_tensor_from((Party.party_id + 2) % 3)
check_is_all_element_equal(pi1, o_pi1)


Party.send_ring_tensor_to((Party.party_id + 1) % 3, pi2)
o_pi2 = Party.receive_ring_tensor_from((Party.party_id + 1) % 3)
check_is_all_element_equal(pi2, o_pi2)




print('finish')





