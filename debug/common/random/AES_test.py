from common.random.kernel.aes import AES
import torch
from config.base_configs import data_type

seed = torch.tensor([2, 2, 5], device='cpu', dtype=data_type)

aes = AES(seed)
bit_len = 128 * 2

# num_of_gen = bit_len // 64

random_num = aes.bit_random(bit_len)

print(random_num)


# import torch
# import torchcsprng as csprng
#
# input = torch.tensor([1], dtype=torch.int64)
# out = torch.empty(2, dtype=torch.int64)
# key = torch.ones([2], dtype=torch.int64)
#
# csprng.encrypt(input, out, key, "aes128", "ecb")
# print(out)
