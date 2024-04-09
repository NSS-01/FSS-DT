from application.NN.NNCS import NeuralNetworkCS
from application.NN.layers.complex_nonlinear_layers import *

from config.base_configs import *
from crypto.tensor.ArithmeticSharedRingTensor import ArithmeticSharedRingTensor

client = NeuralNetworkCS(type='client')
client.inference_transformer(True)
client.connect(('127.0.0.1', 20000), ('127.0.0.1', 20001), ('127.0.0.1', 8089), ('127.0.0.1', 8088))

# k1 = client.receive()

x1 = client.receive()
x1 = ArithmeticSharedRingTensor(x1, client)

res = sec_gelu.forward(x1, device=DEVICE)

print(res.restore())
