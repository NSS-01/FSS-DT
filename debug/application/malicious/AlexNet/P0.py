from application.NN.AlexNet.sec_AlexNet import SecAlexNet
from crypto.mpc.malicious_party import Malicious3PCParty
from crypto.protocols.RSS_malicious_subprotocol.protocols import *
import time

party_id = 0

Party = Malicious3PCParty(party_id=party_id)
Party.online()
Party.generate_prg_seed()

net = SecAlexNet()

x = torch.ones([1, 1, 28, 28], device=DEVICE)
x = RingTensor.convert_to_ring(x)

X = share(x, Party)
start0 = time.time()
for _ in range(10):
    start = time.time()
    res = net.forward(X)
    end = time.time()

    res = open(res)

    print("res", res.convert_to_real_field())
    print("time", end - start)

end = time.time()
print("avg_time", (end - start0) / 10)
