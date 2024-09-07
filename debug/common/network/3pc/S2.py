# from crypto.mpc.malicious_party import Malicious3PCParty
from crypto.mpc.malicious_party_new import Malicious3PCParty

party_id = 2

Party = Malicious3PCParty(party_id=party_id)
Party.online()

a = Party.receive_torch_tensor_from(0)
print(a)

Party.close()
