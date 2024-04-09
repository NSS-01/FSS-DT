from crypto.mpc.malicious_party import *
from config.mpc_configs import *
import time
from crypto.protocols.RSS_malicious_subprotocol.protocols import *
id = 1
# 创建一个参与方Party，它的编号是1
Party = Malicious3PCParty(party_id = id)
Party.online()
Party.generate_prg_seed()

# # rand test
# r = rand_with_prg(10, Party)
# print(r)
#
# # open test
# r_p = open(r)
# print(r_p)
#
#
# # coin test
# r_coin = coin(10, Party)
# print(r_coin)
#
# # recon test
# r_recon = recon(r, 0)
# print(r_recon)
#
# r2 = rand_like(r, Party)
# print(r2)

# x = receive_share_from(0, Party)
# y = receive_share_from(0, Party)
#
#
# z = mul(x, y)
# z_o = open(z)
# print(z_o)
#
# c = check_zero(z)
# print(c)

x0 = torch.tensor([1,0])
x1 = torch.tensor([1,0])
x0 = RingTensor.convert_to_ring(x0)
x1 = RingTensor.convert_to_ring(x1)

x = ReplicatedSecretSharing([x0,x1], Party)

a_x = bit_injection(x)

a = open(a_x)

print(a)
