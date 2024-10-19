
import torch

from NssMPC.secure_model.mpc_party import SemiHonestCS
from NssMPC.crypto.primitives import ArithmeticSecretSharing
from NssMPC.common.ring.ring_tensor import RingTensor
n = 8
server = SemiHonestCS(type='server')
server.set_comparison_provider()
server.set_multiplication_provider()
server.online()
a = ArithmeticSecretSharing(RingTensor.convert_to_ring(torch.ones((1,n))),party=server)
b = ArithmeticSecretSharing(RingTensor.convert_to_ring(torch.ones((n,2))),party=server)

y = a.view(n,-1)*b
c = a*a
c_ = c.restore().convert_to_real_field()

v_y = y.restore().convert_to_real_field()
a_ = a.restore().convert_to_real_field()
b_ =b.restore().convert_to_real_field()
server.close()