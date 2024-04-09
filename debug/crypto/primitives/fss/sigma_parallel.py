import time

import torch

from config.base_configs import *
from crypto.primitives.function_secret_sharing.dpf import DPF
from crypto.tensor.RingTensor import RingTensor

number_of_keys = 10000
test_input = torch.randint(0, 10, [number_of_keys])
y = RingTensor.convert_to_ring(test_input)
alpha = RingTensor.convert_to_ring(torch.tensor(5))
beta = RingTensor.convert_to_ring(torch.tensor(1))
if __name__ == '__main__':
    start = time.time()
    K0, K1 = DPF.gen(number_of_keys, alpha, beta)
    end = time.time()
    print(end - start)
    res_dpf_0 = DPF.eval(y, K0, 0, PRG_TYPE)
    res_dpf_1 = DPF.eval(y, K1, 1, PRG_TYPE)
    print(test_input)
    print((res_dpf_0 + res_dpf_1))
