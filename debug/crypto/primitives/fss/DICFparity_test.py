# import the libraries
# import the libraries
import time

import torch
from crypto.primitives.function_secret_sharing.p_dicf import ParityDICF, ParityDICFKey
from crypto.tensor.RingTensor import RingTensor
from config.base_configs import DEVICE

# generate key in offline phase
num_of_keys = 10
down_bound = torch.tensor([3]).to(DEVICE)
upper_bound = torch.tensor([7]).to(DEVICE)

Key0, Key1 = ParityDICFKey.gen(num_of_keys=num_of_keys)

# evaluate x in online phase
# generate some values what we need to evaluate
# x = RingTensor.convert_to_ring(torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
x = RingTensor.random([num_of_keys])#convert_to_ring(torch.randint(-10, 10, [num_of_keys]))
x_shift = Key0.phi.reshape(x.shape) + Key1.phi.reshape(x.shape) - x
print(x)
# online phase
# Party 0:
res_0 = ParityDICF.eval(x_shift=x_shift, key=Key0, party_id=0)

# Party 1:
start = time.time()
# for _ in range(100):
res_1 = ParityDICF.eval(x_shift=x_shift, key=Key1, party_id=1)
# end = time.time()
# print((end - start)/10)

# restore result
res = res_0 ^ res_1
print(res)
print((x.tensor.signbit()==res).sum())