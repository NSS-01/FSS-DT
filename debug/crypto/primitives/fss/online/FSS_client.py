"""
ASS基础功能测试
"""
from config.base_configs import DTYPE, SCALE
from crypto.mpc.semi_honest_party import SemiHonestCS
from crypto.primitives.beaver.beaver_triples import BeaverBuffer
from crypto.tensor.ArithmeticSharedRingTensor import ArithmeticSharedRingTensor

###########################################
client = SemiHonestCS(type='client')
# client.set_dtype('int')
# client.set_scale(1)
client.set_dtype(DTYPE)
client.set_scale(SCALE)
client.set_beaver_provider(BeaverBuffer(client))
client.set_compare_key_provider()
client.beaver_provider.load_param()
client.connect(('127.0.0.1', 20000), ('127.0.0.1', 20001), ('127.0.0.1', 8089), ('127.0.0.1', 8088))
###########################################
num_of_keys = client.receive_tensor().item()
x_1 = client.receive_ring_tensor()
shared_x = ArithmeticSharedRingTensor(x_1, client)

y_1 = client.receive_ring_tensor()
shared_y = ArithmeticSharedRingTensor(y_1, client)

rs = (shared_x >= shared_y)

print(rs.restore())
