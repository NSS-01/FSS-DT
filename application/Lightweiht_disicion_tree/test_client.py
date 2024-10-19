
import torch

from NssMPC.secure_model.mpc_party import SemiHonestCS
from NssMPC.crypto.primitives import ArithmeticSecretSharing
from NssMPC.common.ring.ring_tensor import RingTensor
n =8
client = SemiHonestCS(type='client')
client.set_comparison_provider()
client.set_multiplication_provider()
client.online()
a = ArithmeticSecretSharing(RingTensor.convert_to_ring(torch.ones((1,n))),party=client)
b = ArithmeticSecretSharing(RingTensor.convert_to_ring(torch.ones((n,2))),party=client)
y = a.view(n,-1)*b

c = a*a
c_ = c.restore().convert_to_real_field()

v_y = y.restore().convert_to_real_field()
a_ = a.restore().convert_to_real_field()
b_ =b.restore().convert_to_real_field()
print(f"a:{a_}\n b:{b_} \n a*b = {v_y}")
print(f"a*a:{c_}")

client.close()