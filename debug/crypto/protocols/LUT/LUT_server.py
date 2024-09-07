from math import sin

import torch

from crypto.protocols.look_up.look_up_table import *

from config.base_configs import *
from crypto.mpc.semi_honest_party import SemiHonestCS
from crypto.primitives.beaver.beaver_triples import BeaverBuffer
from crypto.tensor.RingTensor import RingTensor

import torch
import torch.nn as nn

server = SemiHonestCS(type='server')
server.set_beaver_provider(BeaverBuffer(server))
server.set_wrap_provider()
server.set_compare_key_provider()
server.beaver_provider.load_param()
server.connect(('127.0.0.1', 8089), ('127.0.0.1', 8088), ('127.0.0.1', 20000), ('127.0.0.1', 20001))


def func(inputs):
    relu = nn.ReLU()
    gelu = nn.GELU()

    return relu(inputs) - gelu(inputs)


inputs = torch.arange(0, 256, dtype=torch.int64, device=DEVICE)
table = RingTensor.convert_to_ring(func(inputs / 64))
print(table)

# print(table[131072].convert_to_real_field())

# print(sin(2.0))
normal_x = torch.tensor([[1.2], [3.9]], device=DEVICE)
x = RingTensor.convert_to_ring(normal_x)
# print(x)

k0, k1 = look_up_gen(1, 0, 256)

server.send(k1)

x0, x1 = ArithmeticSharedRingTensor.share(x, 2)
server.send(x1)

x0 = ArithmeticSharedRingTensor(x0, server) / 2 ** 10

res = look_up_eval(x0, k0, 0, 256, table)

print(res.restore().convert_to_real_field())

print(func(normal_x))

# print("========y============")
# res_y = ArithmeticSharedRingTensor(y, server)
# yy = res_y.restore().convert_to_real_field()
# print(yy)
# print(yy.sum())
#
# list_y = yy.tolist()
#
# count = 0
# for i in list_y:
#     if i != 0.0:
#         print(i)
#         print(count)
#     count += 1
#
# print("======u======")
# res_u = ArithmeticSharedRingTensor(u, server)
# uu = res_u.restore().convert_to_real_field()
# print(uu)
# print(uu.sum())
#
# list_u = uu.tolist()
#
# count = 0
# for i in list_u:
#     if i != 0.0:
#         print(i)
#         print(count)
#     count += 1
#
# print("======z======")
# res_z = ArithmeticSharedRingTensor(z, server)
# zz = res_z.restore().convert_to_real_field()
# print(zz)
# print(zz.sum())

# list_u = uu.tolist()
#
# count = 0
# for i in list_u:
#     if i != 0.0:
#         print(i)
#         print(count)
#     count += 1
