from crypto.protocols.RSS_malicious_subprotocol.protocols import *
from crypto.mpc.malicious_party import Malicious3PCParty

from crypto.protocols.CMP.verifiable_sigma import *

party_id = 0

Party = Malicious3PCParty(party_id=party_id)
Party.online()
Party.generate_prg_seed()

num = 10000

x = torch.rand([num], device=DEVICE)
y = torch.rand([num], device=DEVICE)
z = x - y
x = z
print(x)
print(x.signbit() + 0)
x = RingTensor.convert_to_ring(x)

X = share(x, Party)

res = open(get_msb(X))
print(res)

print("error num", (res == x.signbit() + 0).sum() - num)
