import time

from crypto.primitives.function_secret_sharing.verifiable_dpf import VerifiableDPF, VerifiableDPFKey, vdpf_fd_eval
from crypto.tensor.RingTensor import RingTensor
from config.base_configs import *

bit_len = 4
number_of_keys = 10

test_input = torch.arange(0, 2 ** bit_len)
print(test_input)
y = RingTensor.convert_to_ring(test_input)
y.bit_len = bit_len

alpha = RingTensor.convert_to_ring(torch.tensor([6]))
alpha.bit_len = bit_len
# alpha = torch.ones_like(test_input) * 5

# alpha = RingTensor.convert_to_ring(alpha)
beta = RingTensor.convert_to_ring(torch.tensor(1))
beta.bit_len = bit_len

K0, K1 = VerifiableDPF.gen(1, alpha, beta)

start = time.time()
res_dpf_0, pi_0 = VerifiableDPF.eval(y, K0, 0)
end = time.time()
print(end - start)
res_dpf_1, pi_1 = VerifiableDPF.eval(y, K1, 1)

print(res_dpf_0 + res_dpf_1)
print(pi_0 == pi_1)

res_0, pi_0 = vdpf_fd_eval(K0, 0, bit_len=4)

res_1, pi_1 = vdpf_fd_eval(K1, 1, bit_len=4)

print(res_0 + res_1)
print(pi_0 == pi_1)

#
# start = time.time()
# res_ppq_0, pi_0 = VerifiableDPF.ppq(y, K0, 0)
# end = time.time()
# print(end - start)
# res_ppq_1, pi_1 = VerifiableDPF.ppq(y, K1, 1)
#
# print(res_ppq_0 ^ res_ppq_1)
# print(pi_0 == pi_1)
