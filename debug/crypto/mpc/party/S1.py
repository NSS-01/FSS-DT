from crypto.mpc.semi_honest_party import *
from debug.crypto.mpc.party.configs import *
import time
from config.base_configs import SCALE

# 创建一个参与方Party1，它的编号是1，它的总参与方数是3
Party1 = SemiHonestMPCParty(1, 3)
Party1.set_scale(SCALE)
Party1.set_dtype('float')
Party1.start_server(S1_ADDRESS, S1_PORT)

mapping = {0: (S0_ADDRESS, S0_client_port_to_S1), 2: (S2_ADDRESS, S2_client_port_to_S1)}
Party1.set_target_client_mapping(mapping)

# 参与方Party1与其他两方进行链接
Party1.connect_to_other(0, S1_ADDRESS, S1_client_port_to_S0, S0_ADDRESS, S0_PORT)
Party1.connect_to_other(2, S1_ADDRESS, S1_client_port_to_S2, S2_ADDRESS, S2_PORT)

time.sleep(10)

# 创建一个信息，发送给其他两方
b = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1])

Party1.send_torch_tensor_to(0, b)
Party1.send_torch_tensor_to(2, b)

a = Party1.receive_torch_tensor_from(0)
print("receive np array from S0:", 0, "is:", a)

c = Party1.receive_torch_tensor_from(2)
print("receive np array from S2:", 2, "is:", c)

Party1.print_communication_stats()
print("---------------------------------------------------------------------------------------")

# 创建一个Ring_tensor,发送给其他两方
b = torch.tensor([-2.2, -2.2, -2.2, 2.2, 2.2, 2.2, 2.2])
b = RingTensor.convert_to_ring(b)

Party1.send_ring_tensor_to(0, b)
Party1.send_ring_tensor_to(2, b)

a = Party1.receive_ring_tensor_from(0)
print("receive ring tensor from S0:", 0, "is:", a.convert_to_real_field())

c = Party1.receive_ring_tensor_from(2)
print("receive ring tensor from S2:", 2, "is:", c.convert_to_real_field())

Party1.print_communication_stats()