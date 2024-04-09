from crypto.mpc.semi_honest_party import *
from debug.crypto.mpc.party.configs import *
import time

# 创建一个参与方Party0，它的编号是0，它的总参与方数是3
Party0 = SemiHonestMPCParty(0, 3)
Party0.set_scale(SCALE)
Party0.set_dtype("float")
Party0.start_server(S0_ADDRESS, S0_PORT)

mapping = {1: (S1_ADDRESS, S1_client_port_to_S0), 2: (S2_ADDRESS, S2_client_port_to_S0)}
Party0.set_target_client_mapping(mapping)

# 参与方Party0与其他两方进行链接
Party0.connect_to_other(1, S0_ADDRESS, S0_client_port_to_S1, S1_ADDRESS, S1_PORT)
Party0.connect_to_other(2, S0_ADDRESS, S0_client_port_to_S2, S2_ADDRESS, S2_PORT)

time.sleep(10)

# 创建一个信息，发送给其他两方
a = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0])
Party0.send_torch_tensor_to(1, a)
Party0.send_torch_tensor_to(2, a)

b = Party0.receive_torch_tensor_from(1)
print("receive np array from S1:", 1, "is:", b)

c = Party0.receive_torch_tensor_from(2)
print("receive np array from S2:", 2, "is:", c)

Party0.print_communication_stats()
print("---------------------------------------------------------------------------------------")


# 创建一个Ring_tensor,发送给其他两方
a = torch.tensor([-1.1, -1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1])
a = RingTensor.convert_to_ring(a)
Party0.send_ring_tensor_to(1, a)
Party0.send_ring_tensor_to(2, a)

b = Party0.receive_ring_tensor_from(1)
print("receive ring tensor from S1:", 1, "is:", b.convert_to_real_field())

c = Party0.receive_ring_tensor_from(2)
print("receive ring tensor from S2:", 2, "is:", c.convert_to_real_field())

Party0.print_communication_stats()