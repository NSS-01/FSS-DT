import torch
from common.random.prg import PRG
from config.base_configs import LAMBDA, DEVICE
from crypto.tensor.RingTensor import RingTensor


def convert(num, k):
    res = num % (2 ** k)
    return res


# Test PRG with MT kernel on CPU
prg = PRG(kernel='TMT', device='cuda')

num_of_keys = 6
#tmp_0 = torch.tensor([random.randint(0, 1000)], device=DEVICE)
tmp_0 = torch.randint(1,1000,[num_of_keys], device=DEVICE)
prg.set_seeds(tmp_0)
s_0_0 = prg.bit_random(LAMBDA)


tmp_1= torch.randint(1,1000,[num_of_keys], device=DEVICE)
prg.set_seeds(tmp_1)
s_0_1 = prg.bit_random(LAMBDA)

t0 = torch.tensor([0] * num_of_keys, device=DEVICE)
t1 = torch.tensor([1] * num_of_keys, device=DEVICE)


v_a = torch.tensor(0, device=DEVICE)
s_last_0 = s_0_0
s_last_1 = s_0_1

s_l_0, v_l_0, t_l_0, s_r_0, v_r_0, t_r_0 = prg.gen_dcf_keys(s_last_0[:, 0], LAMBDA)
s_l_1, v_l_1, t_l_1, s_r_1, v_r_1, t_r_1 = prg.gen_dcf_keys(s_last_1[:, 0], LAMBDA)
print(s_l_0)
print(s_r_0)


alpha = [5,3,4,1,4,5]
T = RingTensor.convert_to_ring(torch.tensor(alpha))
w = T.get_bit(1)
# 将w转换为列向量
w = w.unsqueeze(1)

for t in T:
    print(t)
