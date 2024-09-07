from crypto.mpc.semi_honest_party import *
from debug.crypto.mpc.party.configs import *
import time
from config.base_configs import SCALE
# 创建一个参与方Party2，它的编号是2，它的总参与方数是3
Party2 = SemiHonestMPCParty(2, 3)
Party2.set_scale(SCALE)
Party2.set_dtype('float')
Party2.start_server(S2_ADDRESS, S2_PORT)

mapping = {0: (S0_ADDRESS, S0_client_port_to_S2), 1: (S1_ADDRESS, S1_client_port_to_S2)}

Party2.set_target_client_mapping(mapping)


# 参与方Party2与其他两方进行链接
Party2.connect_to_other(0, S2_ADDRESS, S2_client_port_to_S0, S0_ADDRESS, S0_PORT)
Party2.connect_to_other(1, S2_ADDRESS, S2_client_port_to_S1, S1_ADDRESS, S1_PORT)

time.sleep(10)

# 创建一个信息，发送给其他两方
c = torch.tensor([2, 2, 2, 2, 2, 2, 2, 2])
Party2.send_torch_tensor_to(0, c)
Party2.send_torch_tensor_to(1, c)

a = Party2.receive_torch_tensor_from(0)
print("receive np array from S0:", 0, "is:", a)
b = Party2.receive_torch_tensor_from(1)
print("receive np array from S1:", 1, "is:", b)

Party2.print_communication_stats()
print("---------------------------------------------------------------------------------------")

# 创建一个Ring_tensor,发送给其他两方
c = torch.tensor([-3.3, -3.3, -3.3, 3.3, 3.3, 3.3, 3.3])
c = RingTensor.convert_to_ring(c)

Party2.send_ring_tensor_to(0, c)
Party2.send_ring_tensor_to(1, c)

a = Party2.receive_ring_tensor_from(0)
print("receive ring tensor from S0:", 0, "is:", a.convert_to_real_field())

b = Party2.receive_ring_tensor_from(1)
print("receive ring tensor from S1:", 1, "is:", b.convert_to_real_field())

Party2.print_communication_stats()


