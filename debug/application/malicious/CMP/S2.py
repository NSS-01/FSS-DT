from crypto.protocols.RSS_malicious_subprotocol.protocols import *
from crypto.mpc.malicious_party import Malicious3PCParty

party_id = 2

Party = Malicious3PCParty(party_id=party_id)
Party.online()
Party.generate_prg_seed()

X = receive_share_from(input_id=0, party=Party)

print(open(get_msb(X)))
