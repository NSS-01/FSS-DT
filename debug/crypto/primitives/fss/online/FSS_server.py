"""
ASS基础功能测试
"""
import time

import torch

from config.base_configs import SCALE, DTYPE
from crypto.mpc.semi_honest_party import SemiHonestCS
from crypto.primitives.beaver.beaver_triples import BeaverBuffer
from crypto.tensor.ArithmeticSharedRingTensor import ArithmeticSharedRingTensor
from crypto.tensor.RingTensor import RingTensor

##########################################
server = SemiHonestCS(type='server')
# server.set_dtype('int')
# server.set_scale(1)
server.set_dtype(DTYPE)
server.set_scale(SCALE)
server.set_beaver_provider(BeaverBuffer(server))
server.set_compare_key_provider()
server.beaver_provider.load_param()
server.connect(('127.0.0.1', 8089), ('127.0.0.1', 8088), ('127.0.0.1', 20000), ('127.0.0.1', 20001))
###########################################
number_of_keys = 10
# 打开下面代码生成FSS密钥
s = time.time()

server.send_tensor(torch.tensor(number_of_keys))

x = torch.rand([number_of_keys], device='cuda:0')
y = torch.rand([number_of_keys], device='cuda:0')

x = RingTensor.convert_to_ring(x)
y = RingTensor.convert_to_ring(y)

print(x >= y)
x_0, x_1 = ArithmeticSharedRingTensor.share(x, 2)
server.send_ring_tensor(x_1)
share_x = ArithmeticSharedRingTensor(x_0, server)

y_0, y_1 = ArithmeticSharedRingTensor.share(y, 2)
server.send_ring_tensor(y_1)
share_y = ArithmeticSharedRingTensor(y_0, server)

rs = (share_x >= share_y)
e = time.time()
print(e - s)
C = rs.restore()
print(C)
