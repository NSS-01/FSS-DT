"""
ASS基础功能测试
"""

from config.base_configs import *
from crypto.mpc.semi_honest_party import SemiHonestCS
from crypto.tensor.ArithmeticSharedRingTensor import ArithmeticSharedRingTensor
from crypto.primitives.beaver.beaver_triples import BeaverBuffer
from crypto.tensor.RingTensor import RingTensor

client = SemiHonestCS(type='client')
client.set_beaver_provider(BeaverBuffer(client))
# client.set_wrap_provider()
client.set_compare_key_provider()
client.set_neg_exp_provider()
client.set_exp_provider()
client.beaver_provider.load_param()
client.connect(('127.0.0.1', 20000), ('127.0.0.1', 20001), ('127.0.0.1', 8089), ('127.0.0.1', 8088))

x_1 = client.receive()
shared_x = ArithmeticSharedRingTensor(x_1, client)

y_1 = client.receive()
shared_y = ArithmeticSharedRingTensor(y_1, client)
client.receive()

# 打开代码测试相应功能
# print("============================================")
# print('shared x', shared_x)
# print('restored x: ', shared_x.restore().convert_to_real_field())
# print("============================================")
# print('shared y', shared_y)
# print('restored y: ', shared_y.restore().convert_to_real_field())
# print("============================================")
# share_z = shared_x + shared_y
# print(share_z.restore())
# print(share_z.restore().convert_to_real_field())
print("==============================================")
share_z = shared_x * shared_y
# print(share_z)
restored_z = share_z.restore()

print('restored z: ', restored_z)
# c = RingTensor.convert_to_ring(torch.ones(size=[2, 2], device=DEVICE))
# share_z = shared_x @ c
# print('shared z: ', share_z)
# print('z: ', share_z.restore().convert_to_real_field())
# print("============================================")
# share_z = shared_x / 8
# print(share_z.restore().convert_to_real_field())
# print("============================================")
# share_z2 = shared_x @ shared_y
# print('shared z2: ', share_z2)
# print('z2: ', share_z2.restore().convert_to_real_field())
print("============================================")
share_z = shared_x >= shared_y
print('share z: ', share_z)
print('restored z: ', share_z.restore().convert_to_real_field())
# print("============================================")
# share_z = shared_x <= shared_y
# print('share z: ', share_z)
# print('restored z: ', share_z.restore().convert_to_real_field())
# print("============================================")
# share_z = shared_x > shared_y
# print('share z: ', share_z)
# print('restored z: ', share_z.restore().convert_to_real_field())
# print("============================================")
# share_z = shared_x < shared_y
# print('share z: ', share_z)
# print('restored z: ', share_z.restore().convert_to_real_field())
# print("============================================")
# share_z = shared_x-RingTensor.convert_to_ring(1)
# shared_x>0
#
# print('restored z: ', share_z.restore().convert_to_real_field())
# print("==============================================")
# share_z = shared_x@RingTensor(torch.ones_like(shared_x.tensor))
# restored_z = share_z.restore()
#
# print('restored z: ', restored_z)

# print("============================================")
# max_x = ArithmeticSharedRingTensor.max(shared_x)
# print(max_x.restore().convert_to_real_field())
# print("============================================")
# share_z = shared_x / shared_y
# print(share_z.restore().convert_to_real_field())
# print("============================================")
# share_z = ArithmeticSharedRingTensor.reciprocal_sqrt(shared_x)
# print(share_z.restore().convert_to_real_field())
client.close()
