from application.NN.NNCS import NeuralNetworkCS
from application.NN.layers.complex_nonlinear_layers import *

from config.base_configs import *
from crypto.tensor.ArithmeticSharedRingTensor import ArithmeticSharedRingTensor
from crypto.tensor.RingTensor import RingTensor

import torch
import torch.nn as nn

server = NeuralNetworkCS(type='server')
server.inference_transformer(True)
server.connect(('127.0.0.1', 8089), ('127.0.0.1', 8088), ('127.0.0.1', 20000), ('127.0.0.1', 20001))

normal_x = torch.tensor([[1.2], [5], [-1.2], [-5]], device=DEVICE)
x = RingTensor.convert_to_ring(normal_x)

x0, x1 = ArithmeticSharedRingTensor.share(x, 2)
server.send(x1)

x0 = ArithmeticSharedRingTensor(x0, server)

res = sec_gelu.forward(x0, device=DEVICE)

print(res.restore().convert_to_real_field())

print(nn.GELU()(normal_x))
