from crypto.mpc.semi_honest_party import *
from config.mpc_configs import *

from crypto.primitives.arithmetic_secret_sharing_dev import *
import time
id = 2
mapping = {0: (S0_ADDRESS, S0_client_port_to_S2), 1: (S1_ADDRESS, S1_client_port_to_S2)}


# 创建一个参与方Party，它的编号是2，它的总参与方数是3
Party = SemiHonestMPCParty(id = id, parties_num = 3)
Party.set_scale(SCALE)
Party.set_dtype(DTYPE)
Party.start_server(S2_ADDRESS, S2_PORT)
Party.set_target_client_mapping(mapping)

mask_path = './data/mask_data/S2/'
Party.load_pre_generated_mask(mask_path)
# print(Party.pre_generated_mask.restore().convert_to_real_field())
# 参与方Party1与其他两方进行链接
Party.connect_to_other(0, S2_ADDRESS, S2_client_port_to_S0, S0_ADDRESS, S0_PORT)
Party.connect_to_other(1, S2_ADDRESS, S2_client_port_to_S1, S1_ADDRESS, S1_PORT)

Party.check_all_parties_online()


shared_tensor = Party.receive_iss_from(0)
print(shared_tensor.restore().convert_to_real_field())


shared_tensor2 = Party.receive_iss_from(0)
print(shared_tensor2.restore().convert_to_real_field())

a_sl = shared_tensor[1]
print(a_sl.restore().convert_to_real_field())
#
# add_result = shared_tensor + shared_tensor
# res = add_result.restore()
#
# print(res)
#
#
mul_result = shared_tensor * 2
res = mul_result.restore()
print(res.convert_to_real_field())

mul_result2 = mul_result * shared_tensor2
res = mul_result2.restore()
print(res.convert_to_real_field())
#
#
# sum_result = shared_tensor.sum(0)
# res = sum_result.restore()
# print(res.convert_to_real_field())
#
# o = iss_DICF(shared_tensor)
# print(o.restore().convert_to_real_field())
#
# ge_result = shared_tensor >= shared_tensor2
# res = ge_result.restore()
#
# print(res.convert_to_real_field())
#
# lt_result = shared_tensor <= shared_tensor2
# res = lt_result.restore()
#
# print(res.convert_to_real_field())

ID = Party.receive_iss_from(0)
print(ID.restore().convert_to_real_field())

u = shared_tensor[0]
print(u.restore().convert_to_real_field())

ID[:,1] = u
print(ID.restore().convert_to_real_field())