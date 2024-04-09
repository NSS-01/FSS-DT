from crypto.tensor.RingTensor import RingTensor
from crypto.primitives.function_secret_sharing.verifiable_p_dicf import vpDICF

import torch

x = RingTensor.convert_to_ring(torch.tensor([1, 2, 3 , 5]))
y = RingTensor.convert_to_ring(torch.tensor([1, 3, 2, 3]))

phi, K0, K1 = vpDICF.gen(4)
shift_input = x - y + phi

res_0, pi_0 = vpDICF.cmp(shift_input, K0, 0)
res_1, pi_1 = vpDICF.cmp(shift_input, K1, 1)

print(res_0 ^ res_1)

print(pi_0 == pi_1)

