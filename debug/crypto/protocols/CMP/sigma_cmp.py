import time

import torch
from crypto.protocols.CMP.cmp_sigma import CMPSigma
from crypto.tensor.RingTensor import RingTensor
from config.base_configs import DEVICE

# generate key in offline phase
num_of_keys = 10

Key0, Key1 = CMPSigma.gen(num_of_keys=num_of_keys)

# evaluate x in online phase
# generate some values what we need to evaluate
# x = RingTensor.convert_to_ring(torch.tensor([-1, 0, 3, 4, 5, 6, 7, 8, 9, 10]))
x = RingTensor.convert_to_ring(torch.randint(0, 10, [num_of_keys]))
y = RingTensor.convert_to_ring(torch.randint(0, 10, [num_of_keys]))
print(x - y)
x = x - y
x_shift = x + Key0.r_in.reshape(x.shape) + Key1.r_in.reshape(x.shape)

# online phase
# Party 0:
res_0 = CMPSigma.eval(party_id=0, keys=Key0, x_shift=x_shift)

# Party 1:
res_1 = CMPSigma.eval(party_id=1, keys=Key1, x_shift=x_shift)

print(res_0 ^ res_1)
