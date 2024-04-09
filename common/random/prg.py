import math
import random

import torch

import common.random.kernel.aes as AES
import common.random.kernel.mt as MT
import common.random.kernel.exp as TMT

from config.base_configs import BIT_LEN, DEVICE, data_type


# TODO:种子混乱
# TODO： 这里是否考虑支持正常的random生成方式
# 记录一下，首先根据并行的种子数量生成以in64为单位的n个数，然后在进行分割
class PRG(object):
    """
    种子并行伪随机数生成器，用于生成伪随机数，可以使用random库，pytorch库，MT库，TMT库等生成随机数。
    """

    def __init__(self, kernel='random', device='cpu'):
        if kernel not in ['random', 'pytorch', 'MT', 'TMT', 'AES']:
            raise KeyError(kernel, " kernel is not supported! It should be one of the random/pytorch/MT")

        self.kernel = kernel
        self.bit_seeds = None  # 用于并行化生成随机数的种子
        self.device = device
        self.shape = None  # 和seed保持一致，用于对随机数进行reshape
        self.len = None

        self.value_random_kernel = 'pytorch'

        self.value_seed = None  # 用于生成随机数的种子

    def set_seeds(self, seeds):
        """
        为PRG设置种子

        此PRG的种子是并行化的，即可以同时生成多个种子对应的多个伪随机数，这里的seeds是一个tensor，tensor内的每一个元素都是一个种子。

        :param seeds: 这里的seeds是一个tensor，tensor内的每一个元素都是一个种子
        :return:

        """
        self.bit_seeds = seeds
        self.shape = seeds.shape
        # 更新算法中这里不能flatten
        # TODO 检查其他算法
        # self.bit_seeds = seeds.flatten().to(self.device)
        self.bit_seeds = seeds.to(self.device)
        self.len = len(self.bit_seeds)

    def set_seed(self, seed):
        self.value_seed = seed

    def bit_random_with_random_kernel(self, bits):
        """ 使用random库生成随机数

        调用python自带的random库生成随机数，随机数具有bits个比特，因为random库本身不支持并行化，所以这里的并行化是通过循环实现的。
        这里的随机数与其他方法生成的不一样，直接指定了位数，因此并不能直接转换为torch.int64, 但是可以直接用于FSS的计算。

        :param bits: 生成随机数的位数
        :return: r: 生成的随机数， 一个list，list内的每一个元素都是一个bits位的随机数

        :raise: ValueError, 如果没有设置种子，就会报错
        """
        # 这特殊一点，因为可以直接生成为n bit的随机数
        # 循环对每个种子生成长度为bits个比特的随机数
        if self.bit_seeds is None:
            raise ValueError("seeds is None, please set seeds first!")
        r = []
        for seed in self.bit_seeds:
            random.seed(seed)
            r.append(random.getrandbits(bits))
        return r

    def bit_random_with_pytorch_kernel(self, bits):
        """ 使用pytorch库生成随机数

        调用pytorch自带的库生成随机数，随机数具有bits个比特，因为pytorch库本身不支持并行化，所以这里的并行化是通过循环实现的。
        不同与random库，这里的torch的random函数只能生成64位的随机数，对于大于64位的随机数，用多个随机数来填充位数

        :param bits: 生成随机数的位数
        :return: r: 生成的随机数，一个tensor，有两个维度，第一个维度是并行化的数量，第二个维度生成bits位随机数随需要的int64的个数

        :raise: ValueError, 如果没有设置种子，就会报错
        """

        if self.bit_seeds is None:
            raise ValueError("seeds is None, please set seeds first!")
        num = math.ceil(bits / BIT_LEN)
        r = torch.empty([self.len, num], dtype=data_type)
        for i, seed in enumerate(self.bit_seeds):
            torch.manual_seed(seed)
            r[i, :] = torch.empty([num], dtype=data_type).random_()
        r.to(self.device)
        return r

    def bit_random_with_MT_kernel(self, bits):
        """ 使用MT库生成随机数

        此处调用自定义的梅森旋转法随机生成器，其遵循梅森19937算法，但是做了并行化的改进，使其可以支持多种子并行

        :param bits: 生成随机数的位数
        :return: out: MT19937生成的随机数，有两个维度，第一个维度是并行化的数量，第二个维度生成bits位随机数随需要的int64的个数

        :raise: ValueError, 如果没有设置种子，就会报错
        """
        if self.bit_seeds is None:
            raise ValueError("seeds is None, please set seeds first!")
        num = math.ceil(bits / BIT_LEN)
        MT.dev = self.device
        mt_prg = MT.TorchMT19937(self.bit_seeds)
        out = mt_prg.random(num)
        return out

    # TODO： 这种方法虽然快，但是是不是真正的随机分布？
    def bit_random_with_exp_kernel(self, bits):
        """ 使用TMT库生成随机数

        此处调用自定义的简化梅森旋转法随机生成器。

        :param bits: 生成随机数的位数
        :return: out: MT19937生成的随机数，有两个维度，第一个维度是并行化的数量，第二个维度生成bits位随机数随需要的int64的个数

        :raise: ValueError, 如果没有设置种子，就会报错
        """

        if self.bit_seeds is None:
            raise ValueError("seeds is None, please set seeds first!")

        num = math.ceil(bits / BIT_LEN)
        TMT_prg = TMT.TMT(self.bit_seeds)
        out = TMT_prg.random(num)
        return out

    def bit_random_with_AES_kernel(self, bits):
        """
        使用AES方法生成随机数
        调用csprng中的库自动生成随机数
        :param bits: 生成随机数的位数
        :return:
        """
        if self.bit_seeds is None:
            raise ValueError("seeds is None, please set seeds first!")

        # num = math.ceil(bits / BIT_LEN)
        AES_prg = AES.AES(self.bit_seeds)
        out = AES_prg.bit_random(bits)
        return out

    def bit_random(self, n_bits):
        """ 生成 n bit的随机数的接口

        :param n_bits: 生成随机数的位数
        :return: gen: 生成的随机数，出去random方法，生成的都是一个tensor，有两个维度，第一个维度是并行化的数量，
        第二个维度生成bits位随机数随需要的int64的个数。 对于random则是一个list。
        """
        # 这里生成的n bit的随机数，因为目前系统只能生成64位，所以按64位划分，生成多个数补齐n位，最后剩余位数都在最后的数里

        if self.kernel == 'random':
            gen = self.bit_random_with_random_kernel(n_bits)
        elif self.kernel == 'pytorch':
            gen = self.bit_random_with_pytorch_kernel(n_bits)
        elif self.kernel == 'MT':
            gen = self.bit_random_with_MT_kernel(n_bits)
        elif self.kernel == 'TMT':
            # TODO 种子位数不同的情况
            self.set_seeds(self.bit_seeds.view(-1, 2)[:, 0].view(-1))
            gen = self.bit_random_with_exp_kernel(n_bits)
        elif self.kernel == 'AES':
            gen = self.bit_random_with_AES_kernel(n_bits)
            # gen = torch.cat(gen, dim=1)
        else:
            gen = self.bit_random_with_pytorch_kernel(n_bits)
        return gen

    def random_with_random_kernel(self, number_of_values):
        """
        Generate a tensor with shape [N, M] of random values, where N is number of keys and M is number of random
        values.
        :param number_of_values:
        :return: a tensor with shape [N, M] of random values
        """
        if self.value_seed is None:
            raise ValueError("seed is None, please set seed first!")
        # data_type = torch.int64
        # if BIT_LEN == 32:
        #     data_type = torch.int32
        # r = torch.empty([number_of_values], dtype=data_type)
        random.seed(self.value_seed)
        r = torch.empty([number_of_values], dtype=data_type).random_()
        r.to(self.device)
        self.value_seed += 1
        return r

    def random_with_torch_kernel(self, number_of_values):
        """
        Generate a tensor with shape [N, M] of random values, where N is number of keys and M is number of random
        values.
        :param number_of_values:
        :return: a tensor with shape [N, M] of random values
        """

        if self.value_seed is None:
            raise ValueError("seed is None, please set seed first!")
        # data_type = torch.int64
        # if BIT_LEN == 32:
        #     data_type = torch.int32

        torch.manual_seed(self.value_seed)
        r = torch.empty([number_of_values], dtype=data_type).random_()
        r.to(self.device)
        self.value_seed += 1

        return r

    # def random_with_MT_kernel(self, number_of_values):
    #     '''
    #     Generate a tensor with shape [N, M] of random values, where N is number of keys and M is number of random
    #     values.
    #     :param number_of_values:
    #     :return: a tensor with shape [N, M] of random values
    #     '''
    #
    #     if self.seeds is None:
    #         raise ValueError("seeds is None, please set seeds first!")
    #
    #     num = math.ceil(number_of_values / 64)
    #     mt_prg = MT19937(self.seeds)
    #     out = mt_prg.random(num)
    #     return out
    #

    # def random_with_AES_kernel(self, number_of_values):
    #     '''
    #     Generate a tensor with shape [N, M] of random values, where N is number of keys and M is number of random
    #     values.
    #     :param number_of_values:
    #     :return: a tensor with shape [N, M] of random values
    #     '''
    #
    #     if self.bit_seeds is None:
    #         raise ValueError("seeds is None, please set seeds first!")
    #
    #     AES_prg = AES.AES(self.bit_seeds)
    #     out = AES_prg.bit_random(number_of_values)
    #     return out

    def random(self, number_of_values):
        """
        Generate a tensor with shape [N, M] of random values, where N is number of keys and M is number of random
        values.
        :param number_of_values:
        :return: a tensor with shape [N, M] of random values
        """
        if self.value_random_kernel == 'random':
            return self.random_with_random_kernel(number_of_values)
        elif self.value_random_kernel == 'pytorch':
            return self.random_with_torch_kernel(number_of_values)
        # elif self.kernel == 'MT':
        #     return self.bit_random_with_MT_kernel(number_of_values)
        # elif self.kernel == 'TMT':
        #     return self.bit_random_with_exp_kernel(number_of_values)
        # elif self.kernel == 'AES':
        #     # todo: AES重构
        #     return self.random_with_AES_kernel(number_of_values)
        else:
            return self.random_with_torch_kernel(number_of_values)

    def bit_random_for_fss_cw(self, bits):
        # 看AES的注释
        if self.kernel == 'random':
            gen = self.bit_random_with_random_kernel(bits)
        elif self.kernel == 'pytorch':
            gen = self.bit_random_with_pytorch_kernel(bits)
        elif self.kernel == 'MT':
            gen = self.bit_random_with_MT_kernel(bits)
        elif self.kernel == 'AES':
            if self.bit_seeds is None:
                raise ValueError("seeds is None, please set seeds first!")
            AES_prg = AES.AES(self.bit_seeds)
            gen = AES_prg.bit_random_for_fss_cw(bits)
        else:
            gen = self.bit_random_with_MT_kernel(bits)
        return gen
