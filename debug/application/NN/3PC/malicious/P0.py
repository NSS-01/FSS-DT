import torch

from application.NN.AlexNet.sec_AlexNet import SecAlexNet
from crypto.mpc.malicious_party import Malicious3PCParty
# from application.NN.ResNet.sec_ResNet import *
from crypto.mpc.semi_honest_party import *
from config.mpc_configs import *
from config.base_configs import *

from crypto.tensor.RingTensor import RingTensor
from crypto.primitives.arithmetic_secret_sharing import *
from crypto.primitives.arithmetic_secret_sharing.replicated_secret_sharing import ReplicatedSecretSharing

import time

party_id = 0
# 创建一个参与方Party，它的编号是0，它的总参与方数是3
Party = Malicious3PCParty(party_id=party_id)
Party.online()
Party.generate_prg_seed()
Party.os_cmp.preprocess(10)
Party.os_dpf.preprocess(10)
Party.beaver_provider.load_triples(3)
Party.beaver_provider_2pc.load_triples()

net = SecAlexNet()
# net = resnet50()

x = torch.ones([1, 1, 28, 28], dtype=torch.float64, device=DEVICE)
x = RingTensor.convert_to_ring(x)

X = ReplicatedSecretSharing.gen_and_share(x, Party)

Party.receive_torch_tensor_from(1)
Party.receive_torch_tensor_from(2)

total = 0
for _ in range(11):
    start = time.time()
    res = net.forward(X)
    Party.os_cmp.check_part.check()
    Party.os_cmp.check_part.clear()
    end = time.time()

    res = res.restore()

    print("res", res.convert_to_real_field())
    print("time", end - start)
    if _ != 0:
        total += end - start

print("avg_time", total / 10)
