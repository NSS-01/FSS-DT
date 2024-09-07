import time

import torch

from config.base_configs import *
from crypto.primitives.function_secret_sharing.dcf import DCF, DCFKey
from crypto.tensor.RingTensor import RingTensor

number_of_keys = 100
test_input = torch.randint(0, 10, [number_of_keys], device=DEVICE)
y = RingTensor.convert_to_ring(test_input)
alpha = RingTensor.convert_to_ring(torch.tensor(5,device=DEVICE))
beta = RingTensor.convert_to_ring(torch.tensor(1,device=DEVICE))
if __name__ == '__main__':
    start = time.time()
    K0, K1 = DCFKey.gen(number_of_keys, alpha, beta)

    res_dcf_0 = DCF.eval(y, K0, 0, PRG_TYPE)
    res_dcf_1 = DCF.eval(y, K1, 1, PRG_TYPE)
    print(test_input)
    print((res_dcf_0 + res_dcf_1))
    end = time.time()
    print("DCF密钥数量", number_of_keys)
    print("生成时间", end - start, "秒")
