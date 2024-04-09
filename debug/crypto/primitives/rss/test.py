import torch

from crypto.mpc.semi_honest_party import *
from config.mpc_configs import *
from crypto.tensor.RingTensor import *
from crypto.primitives.arithmetic_secret_sharing.replicated_secret_sharing import *
from application.clustering.PLE_DBSCAN.protocols.edc_parallelization import *

party_id = 0
# dataset_name = 'aggregation'

mapping = {1: (S1_ADDRESS, S1_client_port_to_S0), 2: (S2_ADDRESS, S2_client_port_to_S0)}

# 创建一个参与方Party，它的编号是0，它的总参与方数是3
Party = SemiHonestMPCParty(id=party_id, parties_num=3)
Party.set_scale(SCALE)
Party.set_dtype(DTYPE)
Party.start_server(S0_ADDRESS, S0_PORT)
Party.set_target_client_mapping(mapping)

# Party.connect_to_other(1, S0_ADDRESS, S0_client_port_to_S1, S1_ADDRESS, S1_PORT)
# Party.connect_to_other(2, S0_ADDRESS, S0_client_port_to_S2, S2_ADDRESS, S2_PORT)
#
# Party.check_all_parties_online()


# r = RingTensor.convert_to_ring(torch.tensor(5))
#
# r0,r1,r2 = ReplicatedSecretSharing.share(r)
#
# r = ReplicatedSecretSharing(r0,party=None)
#
# r.save('./')
#
# r_file = r.load_from_file('./')
#
# print("r0",r.replicated_shared_tensor[0])
#
# print("r_file",r_file.replicated_shared_tensor[0])

r = RandomRSSProvider()

r.gen_random_rss(6)

# r_file = r.load(6, Party)

r_get = r.get(6)

print("r_get",r_get)

r_get1 = r.get(1)
print("r_get1",r_get1)