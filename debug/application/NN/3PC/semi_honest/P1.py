from application.NN.AlexNet.sec_AlexNet import SecAlexNet
# from application.NN.ResNet.sec_ResNet import *
from crypto.mpc.semi_honest_party import *
from config.mpc_configs import *
from config.base_configs import *
import time

from crypto.primitives.arithmetic_secret_sharing.replicated_secret_sharing import ReplicatedSecretSharing

id = 1
mapping = {0: (S0_ADDRESS, S0_client_port_to_S1), 2: (S2_ADDRESS, S2_client_port_to_S1)}

# 创建一个参与方Party，它的编号是1，它的总参与方数是3
Party = SemiHonestMPCParty(id=id, parties_num=3)
Party.set_scale(SCALE)
Party.set_dtype(DTYPE)
Party.start_server(S1_ADDRESS, S1_PORT)
Party.set_target_client_mapping(mapping)

# 参与方Party1与其他两方进行链接
Party.connect_to_other(0, S1_ADDRESS, S1_client_port_to_S0, S0_ADDRESS, S0_PORT)
Party.connect_to_other(2, S1_ADDRESS, S1_client_port_to_S2, S2_ADDRESS, S2_PORT)

Party.check_all_parties_online()

Party.generate_prg_seed()

net = SecAlexNet()
# net = resnet50()

X = Party.receive_rss_from(0)

for _ in range(10):
    res = net.forward(X)

    res = res.restore()
    print(res)
