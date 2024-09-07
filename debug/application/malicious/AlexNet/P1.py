from application.NN.AlexNet.sec_AlexNet import SecAlexNet
from crypto.mpc.malicious_party import Malicious3PCParty
from crypto.protocols.RSS_malicious_subprotocol.protocols import *

party_id = 1

Party = Malicious3PCParty(party_id=party_id)
Party.online()
Party.generate_prg_seed()

net = SecAlexNet()

X = receive_share_from(input_id=0, party=Party)

for _ in range(10):
    res = net.forward(X)

    res = open(res)
    print(res)
