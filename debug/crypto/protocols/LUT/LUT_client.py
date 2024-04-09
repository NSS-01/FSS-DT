from math import sin

from crypto.protocols.look_up.look_up_table import *

from config.base_configs import *
from crypto.mpc.semi_honest_party import SemiHonestCS
from crypto.tensor.ArithmeticSharedRingTensor import ArithmeticSharedRingTensor
from crypto.primitives.beaver.beaver_triples import BeaverBuffer
from crypto.tensor.RingTensor import RingTensor

import torch
import torch.nn as nn
import torch.nn.functional as F

client = SemiHonestCS(type='client')
client.set_beaver_provider(BeaverBuffer(client))
client.set_wrap_provider()
client.set_compare_key_provider()
client.beaver_provider.load_param()
client.connect(('127.0.0.1', 20000), ('127.0.0.1', 20001), ('127.0.0.1', 8089), ('127.0.0.1', 8088))


def func(inputs):
    relu = nn.ReLU()
    gelu = nn.GELU()

    return relu(inputs) - gelu(inputs)


inputs = torch.arange(0, 256, dtype=torch.int64, device=DEVICE)
table = RingTensor.convert_to_ring(func(inputs / 64))

print(torch.unique(table.convert_to_real_field()).shape)
k1 = client.receive()

x1 = client.receive()
x1 = ArithmeticSharedRingTensor(x1, client) / 2 ** 10

res = look_up_eval(x1, k1, 0, 256, table)

print(res.restore())
#
# res_y = ArithmeticSharedRingTensor(y, client)
# print(res_y.restore())
#
#
# res_u = ArithmeticSharedRingTensor(u, client)
# print(res_u.restore())
#
# res_z = ArithmeticSharedRingTensor(z, client)
# print(res_z.restore())
