from application.NN.AlexNet.sec_AlexNet import SecAlexNet
from crypto.mpc.malicious_party import Malicious3PCParty
# from application.NN.ResNet.sec_ResNet import *
from crypto.mpc.semi_honest_party import *
from config.mpc_configs import *
from config.base_configs import *
import time

from crypto.primitives.arithmetic_secret_sharing.replicated_secret_sharing import ReplicatedSecretSharing

party_id = 1
Party = Malicious3PCParty(party_id=party_id)

Party.online()
Party.generate_prg_seed()
Party.os_cmp.preprocess(10)
Party.os_dpf.preprocess(10)
Party.beaver_provider.load_triples(3)
Party.beaver_provider_2pc.load_triples()

net = SecAlexNet()
# net = resnet50()

X = Party.receive_rss_from(0)

Party.send_torch_tensor_to(0, torch.tensor(0))

for _ in range(11):
    res = net.forward(X)
    Party.os_cmp.check_part.check()
    Party.os_cmp.check_part.clear()

    res = res.restore()
    print(res)
