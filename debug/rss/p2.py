from crypto.mpc.semi_honest_party import *
from config.mpc_configs import *
from config.base_configs import *
import time

id = 2
mapping = {0: (S0_ADDRESS, S0_client_port_to_S2), 1: (S1_ADDRESS, S1_client_port_to_S2)}

# 创建一个参与方Party，它的编号是2，它的总参与方数是3
Party = SemiHonestMPCParty(id=id, parties_num=3)
Party.set_scale(SCALE)
Party.set_dtype(DTYPE)
Party.start_server(S2_ADDRESS, S2_PORT)
Party.set_target_client_mapping(mapping)

# 参与方Party1与其他两方进行链接
Party.connect_to_other(0, S2_ADDRESS, S2_client_port_to_S0, S0_ADDRESS, S0_PORT)
Party.connect_to_other(1, S2_ADDRESS, S2_client_port_to_S1, S1_ADDRESS, S1_PORT)

Party.check_all_parties_online()

shared_tensor = Party.receive_rss_from(0)
# print(shared_tensor)

# add_result = shared_tensor + shared_tensor
# res = add_result.restore()
#
# print(res)

mul_result = shared_tensor * shared_tensor
res = mul_result.restore()
print(res.convert_to_real_field())