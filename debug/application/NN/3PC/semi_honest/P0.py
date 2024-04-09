import time

from application.NN.AlexNet.sec_AlexNet import SecAlexNet
from config.base_configs import *
from config.mpc_configs import *
# from application.NN.ResNet.sec_ResNet import *
from crypto.mpc.semi_honest_party import *
from crypto.primitives.arithmetic_secret_sharing.replicated_secret_sharing import ReplicatedSecretSharing
from crypto.tensor.RingTensor import RingTensor

id = 0
mapping = {1: (S1_ADDRESS, S1_client_port_to_S0), 2: (S2_ADDRESS, S2_client_port_to_S0)}

# 创建一个参与方Party，它的编号是0，它的总参与方数是3
Party = SemiHonestMPCParty(id=id, parties_num=3)
Party.set_scale(SCALE)
Party.set_dtype(DTYPE)
Party.set_compare_key_provider(ParamBuffer(ParityDICFKey, Party))
Party.compare_key_provider.load_keys(3)
Party.start_server(S0_ADDRESS, S0_PORT)
Party.set_target_client_mapping(mapping)

# 参与方Party0与其他两方进行链接
Party.connect_to_other(1, S0_ADDRESS, S0_client_port_to_S1, S1_ADDRESS, S1_PORT)
Party.connect_to_other(2, S0_ADDRESS, S0_client_port_to_S2, S2_ADDRESS, S2_PORT)

Party.check_all_parties_online()

Party.generate_prg_seed()

net = SecAlexNet()
# net = resnet50()

x = torch.ones([1, 1, 28, 28], dtype=torch.float64, device=DEVICE)
x = RingTensor.convert_to_ring(x)

X = ReplicatedSecretSharing.gen_and_share(x, Party)
start0 = time.time()
for _ in range(10):
    start = time.time()
    res = net.forward(X)
    end = time.time()

    res = res.restore()

    print("res", res.convert_to_real_field())
    print("time", end - start)

end = time.time()
print("avg_time", (end - start0) / 10)
