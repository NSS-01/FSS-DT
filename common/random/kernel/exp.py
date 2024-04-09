import torch
from config.base_configs import DEVICE
import random


def _torch_int32(x):
    # lsb = 0xFFFFFFFF & x
    # return lsb
    return x


def _int32(x):
    # Get the 32 least significant bits.
    return x & 0xFFFFFFFF


class TMT:
    def __init__(self, seeds):
        if len(seeds.shape) != 1:
            raise ValueError('seeds 只能是一维数组', )
        self.l = seeds.shape[0]
        self.s = seeds
        self.bias = 1234

    def random(self, num):
        """
        生成64位随机数
        :param num: 每个种子要生成的随机数数量
        :return: 并行数同种子数长度为num的随机数列表
        """
        out = torch.empty([self.l, num], dtype=torch.int64, device=DEVICE)
        cur = self.s + self.bias
        for i in range(num + 1):
            cur = 1812433253 * (cur ^ cur >> 30) + i
            y = cur
            # Right shift by 11 bits
            y = y ^ y >> 11
            # Shift y left by 7 and take the bitwise and of 2636928640
            y = y ^ y << 7 & 2636928640
            # Shift y left by 15 and take the bitwise and of y and 4022730752
            y = y ^ y << 15 & 4022730752
            # Right shift by 18 bits
            y = y ^ y >> 18
            if i > 0:
                out[:, i - 1] = y
            #
            # out[:, i] = y

        return out
