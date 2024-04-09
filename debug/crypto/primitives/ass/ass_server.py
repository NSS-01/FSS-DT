"""
ASS基础功能测试
"""
import time

import torch

from config.base_configs import *
from crypto.mpc.semi_honest_party import SemiHonestCS
from crypto.tensor.ArithmeticSharedRingTensor import ArithmeticSharedRingTensor
from crypto.primitives.beaver.beaver_triples import BeaverBuffer
from crypto.tensor.RingTensor import RingTensor

server = SemiHonestCS(type='server')
server.set_beaver_provider(BeaverBuffer(server))
# server.set_wrap_provider()
server.set_compare_key_provider()
server.set_neg_exp_provider()
server.set_exp_provider()
server.beaver_provider.load_param()
server.connect(('127.0.0.1', 8089), ('127.0.0.1', 8088), ('127.0.0.1', 20000), ('127.0.0.1', 20001))

x = torch.randint(high=100, size=[1000], device=DEVICE)
y = torch.randint(high=100, size=[1000], device=DEVICE)
# x = torch.rand([1, 10000],device=DEVICE)
# y = torch.rand([10000,1],device=DEVICE)
# x = torch.tensor([[0.01, 0.51], [-0.06, 1.2]])
# y = torch.tensor([[0.21, -3.6], [-1.2, 1.2]])
x_ring = RingTensor.convert_to_ring(x)

x_0, x_1 = ArithmeticSharedRingTensor.share(x_ring, 2)
server.send(x_1)
share_x = ArithmeticSharedRingTensor(x_0, server)

y_ring = RingTensor.convert_to_ring(y)

y_0, y_1 = ArithmeticSharedRingTensor.share(y_ring, 2)
server.send(y_1)
share_y = ArithmeticSharedRingTensor(y_0, server)

# server.beaver_provider.gen_matrix_beaver_for_parties(x.shape, y.shape)
# 保证两方通信
server.send(torch.tensor(1))

# 打开代码测试相应功能
# print("============================================")
# print('origin x', x)
# # print('shared x', share_x)
# restored_x = share_x.restore()
# print('restored x: ', restored_x.convert_to_real_field())
# print("============================================")
# print('origin y', y)
# # print('shared y', share_y)
# restored_y = share_y.restore()
# print('restored y: ', restored_y.convert_to_real_field())
# print("============================================")
# print('z=x+y', x + y)
# share_z = share_x + share_y
# print(share_z.restore())
# print(share_z.restore().convert_to_real_field())
print("==============================================")
print('z=x*y: ', x * y)
st = time.time()
share_z = share_x * share_y
et = time.time()
print(f"乘法时间: {et-st}")
# print(share_z)
restored_z = share_z.restore()

print('restored z: ', restored_z)
# print("==============================================")
# share_z = share_x@RingTensor(torch.ones_like(share_x.tensor))
# restored_z = share_z.restore()
#
# print('restored z: ', restored_z)

# # 将数恢复到实数域上
# print('real z: ', restored_z.convert_to_real_field())
# print("============================================")
# print("x/8", x / 8)
# share_z = share_x / 8
# print("restore: ", share_z.restore().convert_to_real_field())
# print("==============================================")
# c = RingTensor.convert_to_ring(torch.ones(size=[2, 2], device=DEVICE))
# print('z=x*y: ', x @ torch.ones(size=[2, 2], device=DEVICE))
# share_z = share_x @ c
# # print(share_z)
# restored_z = share_z.restore()
#
# print('restored z: ', restored_z)
# # 将数恢复到实数域上
# print('real z: ', restored_z.convert_to_real_field())
# print("============================================")
# print('z=x@y', x @ y)
# start = time.time()
# share_z2 = share_x @ share_y
# end = time.time()
# # print('shared z2: ', share_z2)
# restored_z2 = share_z2.restore()
# # print('restored_z2: ', restored_z2)
# # 将数恢复到实数域上
# print('real z2: ', restored_z2.convert_to_real_field())
# print(end - start)
# print("============================================")
print(x >= y)
st = time.time()
share_z = share_x >= share_y
et = time.time()
print(f"比较时间：{et-st}")
# print('share z: ', share_z)
print('restored z: ', share_z.restore().convert_to_real_field())
print("============================================")
# print(x <= y)
# share_z = share_x <= share_y
# # print('share z: ', share_z)
# print('restored z: ', share_z.restore().convert_to_real_field())
# print("============================================")
# print(x > y)
# share_z = share_x > share_y
# # print('share z: ', share_z)
# print('restored z: ', share_z.restore().convert_to_real_field())
# print("============================================")
# share_z = share_x < share_y
# # print('share z: ', share_z)
# print('restored z: ', share_z.restore().convert_to_real_field())
# print("============================================")
# share_z = share_x-RingTensor.convert_to_ring(1)
# print('restored z: ', share_z.restore().convert_to_real_field())
# print("============================================")
# print(x)
# print(torch.max(x, dim=1))
# max_x = ArithmeticSharedRingTensor.max(share_x)
# print(max_x.restore().convert_to_real_field())
# print("============================================")
# print("x / y", x / y)
# start = time.time()
# share_z = share_x / share_y
# end = time.time()
# print(end - start)
# print("restored z", share_z.restore().convert_to_real_field())
# print("============================================")
# print("1/sqrt(x)", 1 / torch.sqrt(x))
# share_z = ArithmeticSharedRingTensor.reciprocal_sqrt(share_x)
# print("restored z", share_z.restore().convert_to_real_field())
server.close()