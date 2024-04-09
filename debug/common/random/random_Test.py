from common.random.prg import PRG
# from common.random.AES import csprng
from config.base_configs import DEVICE
import torch
from config.base_configs import data_type
# TODO:好像还有点不对

# Test PRG with TMT kernel on CPU
prg = PRG(kernel='AES', device='cpu')
# seeds = torch.randint(0, 1000, [2], dtype=torch.int32)
seeds = torch.randint(0, 1000, [10, 4], dtype=data_type)
print(seeds)
prg.set_seeds(seeds)
random_num = prg.bit_random(128 * 2 + 2)
print(random_num)
random_num = prg.bit_random(128 * 2 + 2)
print(random_num.shape)

# random_num = prg.bit_random_with_AES_kernel(128 * 2)
# print(random_num)

# # Test PRG with AES kernel on CPU
#prg = PRG(kernel='pytorch', device=DEVICE)
# seeds = torch.randint(0, 10, [1], dtype=torch.int64)
# seeds = torch.tensor([1,2,1,2], dtype=torch.int64)
# print(seeds)
# prg.set_seed(0)
# random_num = prg.random(5)
# print(random_num)
# random_num = prg.random(5)
# print(random_num)



# Test PRG with MT kernel on CPU
# prg = PRG(kernel='MT', device='cpu')
# seeds = torch.randint(0, 10, [10], dtype=torch.int64)
# print(seeds)
# prg.set_seeds(seeds)
# st = time.time()
# random_num = prg.gen_n_bit_random_number(128)
# et = time.time()
# print(et-st)
# print(random_num)

#
# Test PRG with pytorch kernel on CPU
# prg = PRG(kernel='pytorch', device='cpu')
# seeds = torch.randint(0, 10, [100], dtype=torch.int64)
# print(seeds)
# prg.set_seeds(seeds)
# random_num = prg.gen_N_nit_random_number(128)
# print(random_num)
#
#
# # Test PRG with random kernel on CPU
# prg = PRG(kernel='random', device='cpu')
# seeds = torch.randint(0, 10, [100], dtype=torch.int64)
# print(seeds)
# prg.set_seeds(seeds)
# random_num = prg.gen_N_nit_random_number(128)
# print(random_num)
# code a CNN net for test


# Test PRG random number generator with torch kernel
# prg = PRG(kernel='AES', device=DEVICE)
# seeds = torch.randint(0, 10, [2], dtype=torch.int64)
# print(seeds)
# prg.set_seeds(seeds)
# num = 10
# random_num = prg.random(num)
# print(random_num)


