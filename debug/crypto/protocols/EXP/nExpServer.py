from crypto.protocols.exp.neg_exp import *

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
server.set_neg_exp_provider()
server.beaver_provider.load_param()
server.connect(('127.0.0.1', 8089), ('127.0.0.1', 8088), ('127.0.0.1', 20000), ('127.0.0.1', 20001))

normal_x = torch.tensor([[0.01], [1.2], [17.]], device=DEVICE)
x = RingTensor.convert_to_ring(normal_x)
# print(x)

k0 = server.neg_exp_provider.get_parameters(1)

x0, x1 = ArithmeticSharedRingTensor.share(x, 2)
server.send(x1)

x0 = ArithmeticSharedRingTensor(x0, server)

res = neg_exp_eval(x0, k0)

print(res.restore().convert_to_real_field())

print(torch.exp(-normal_x))
