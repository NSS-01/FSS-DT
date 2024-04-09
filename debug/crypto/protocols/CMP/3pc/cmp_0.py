from application.decision_tree.mostree import compare_threshold
from config.mpc_configs import *
from crypto.mpc.semi_honest_party import SemiHonestMPCParty
from crypto.protocols.RSS_malicious_subprotocol.protocols import *
from crypto.primitives.oblivious_transfer.oblivious_select_dpf import ObliviousSelect
from crypto.primitives.oblivious_transfer.msb_with_os import MSBWithOS

from crypto.protocols.CMP.verifiable_sigma import *
from crypto.protocols.providers.key_provider import DICFProvider

party_id = 0

id = 0
mapping = {1: (S1_ADDRESS, S1_client_port_to_S0), 2: (S2_ADDRESS, S2_client_port_to_S0)}

# 创建一个参与方Party，它的编号是0，它的总参与方数是3
Party = SemiHonestMPCParty(id=id, parties_num=3)
Party.set_scale(SCALE)
Party.set_dtype(DTYPE)
Party.set_compare_key_provider()
Party.compare_key_provider.load_keys(3)
Party.start_server(S0_ADDRESS, S0_PORT)
Party.set_target_client_mapping(mapping)

# 参与方Party0与其他两方进行链接
Party.connect_to_other(1, S0_ADDRESS, S0_client_port_to_S1, S1_ADDRESS, S1_PORT)
Party.connect_to_other(2, S0_ADDRESS, S0_client_port_to_S2, S2_ADDRESS, S2_PORT)

Party.check_all_parties_online()

Party.generate_prg_seed()

num = 10

x = torch.randint(-5, 5, [num], dtype=data_type, device=DEVICE)
print(x)
x = RingTensor.convert_to_ring(x)

y = torch.randint(-5, 5, [num], dtype=data_type, device=DEVICE)
print(y)
y = RingTensor.convert_to_ring(y)

X = ReplicatedSecretSharing.gen_and_share(x, Party)
print(X.restore())
Y = ReplicatedSecretSharing.gen_and_share(y, Party)
print(Y.restore())

Party.receive_torch_tensor_from(1)
Party.receive_torch_tensor_from(2)

start = time.time()
res = X < Y
end = time.time()
open_res = res.restore()

print("time: ", end - start)

print(open_res)
print((open_res.tensor == (x < y)).sum() - num)
