from crypto.primitives.beaver.beaver_triples import BeaverBuffer
from crypto.mpc.malicious_party import Malicious3PCParty

BeaverBuffer.gen_beaver_triples_for_malicious(10000)

provider0 = BeaverBuffer(Malicious3PCParty(party_id=0))
provider1 = BeaverBuffer(Malicious3PCParty(party_id=1))
provider2 = BeaverBuffer(Malicious3PCParty(party_id=2))

provider0.load_param(3)
provider1.load_param(3)
provider2.load_param(3)

a0, b0, c0 = provider0.get_parameters(10)
a1, b1, c1 = provider1.get_parameters(10)
a2, b2, c2 = provider2.get_parameters(10)

print(a0)
print(b0)
print(c0)
print("=====================")

print(a1)
print(b1)
print(c1)
print("=====================")

print(a2)
print(b2)
print(c2)
print("=====================")

print(a0 + a1 + a2)
print(b0 + b1 + b2)
print(c0 + c1 + c2)
print("=====================")
