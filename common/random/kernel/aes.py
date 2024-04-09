import math

import torch
import torchcsprng as csprng

from common.utils.tensor_utils import fast_concat, int64_to_int32x2
from config.base_configs import DEVICE, BIT_LEN, data_type

"""
目的：使用AES算法生成随机数
csprng目前只支持Python3.8.0
PyTorch的加密安全伪随机数生成器
"""


class AES:
    def __init__(self, seeds):
        """
        AES要求每个种子为两个int64(128bit)
        :param seeds: 伪随机数生成的种子
        """
        # if len(seeds.shape) != 2:
        #     raise ValueError('seeds must be a two-dimensional array')
        self.s = seeds

        # 这里的种子应该是两个维度的，第一个维度代表多个并行的种子，第二个维度种子按照BIT_LEN拆分成多个int64或者int32

    def random(self, num):
        """
        每个种子生成num个伪随机数，随BIT_LEN的不同，生成的单个随机数的位宽不同
        :param num: 生成伪随机数的个数
        :return: 伪随机数tensor，第一个维度是并行化的数量（种子个数），第二个维度是用对应种子生成的num个伪随机数
        """
        _num = math.ceil(num / 2) if BIT_LEN == 32 else num
        gen_list = self._random(_num)
        if BIT_LEN == 32:
            gen_list = [int64_to_int32x2(x) for x in gen_list]
        gen = fast_concat(gen_list, dim=1)
        return gen[:, :num]

    def bit_random(self, bits):
        """
        每个种子生成bits位的伪随机数，用int64承载，与BIT_LEN无关
        :param bits: 需要随机数的位宽
        :return: tensor，第一个维度是并行化的数量（种子个数），第二个维度是承载bits位随机数所需要的int64们
        """
        # num = math.ceil(bits / 64)
        random_out = self._random_repeat(bits)
        # random_out = self._random_new(bits)
        return random_out
      #  return fast_concat(gen_list, dim=1)

    def bit_random_for_fss_cw(self, bits):
        """
        为了保证FSS的CW生成速度而留下的较低层的接口，concat对CW而言是不需要的
        :param bits: 需要随机数的位宽
        :return: 生成的随机数tensor
        """
        num = math.ceil(bits / 64)
        return self._random(num)

    def _random(self, num):
        out = torch.empty(self.s.shape[0], dtype=torch.int64, device=DEVICE)
        inputs = self.s.to(torch.int64)
        encrypted_seeds_list = []
        for i in range(math.ceil(num / 2)):
            csprng.encrypt(inputs, out, torch.tensor([i, i], device=DEVICE), "aes128", "ecb")
            inputs = out
            encrypted_seeds_list.append(out.clone().view(-1, 2))
        if num % 2 == 1:
            encrypted_seeds_list[-1] = encrypted_seeds_list[-1][:, 0].unsqueeze(1)
        return encrypted_seeds_list

    # 这个不太准确
    def _random_new(self, bits):
        # 種子大小：
        seed_byte = BIT_LEN // 8
        output_byte = math.ceil(bits // (8 * 16)) * 16

        input_num = output_byte // seed_byte
        output_num = output_byte // seed_byte

        increment = torch.arange(input_num, dtype=data_type).view(1, -1)

        initial = self.s.unsqueeze(1) + increment
        print(initial)
        encryped = torch.empty([self.s.shape[0], output_num], dtype= data_type, device=DEVICE)

        csprng.encrypt(initial, encryped, torch.tensor([1, 3], device=DEVICE), "aes128", "ecb")
        return encryped

    # 循环生成，这个应该满足安全性定义
    def _random_repeat(self, bits):
        shape = list(self.s.shape)
        if len(self.s.shape) != 2:
            self.s = self.s.view(-1, 2)
            # assert ValueError('seeds must be a two-dimensional array')
        # 元素的byte大小
        element_byte = BIT_LEN // 8
        # 种子按照BIT_LEN拆分成多个int64或者int32
        num_of_slice = self.s.shape[1]

        block_num = 128 // BIT_LEN

        if num_of_slice % block_num:
            padding_num = block_num - num_of_slice % block_num
            padding_seed = torch.zeros([self.s.shape[0], padding_num], dtype=self.s.dtype, device=DEVICE)

            self.s = torch.cat([self.s, padding_seed], dim=1)

        num_of_slice = self.s.shape[1]

        # 種子的byte大小：
        seed_byte = element_byte * num_of_slice
        # 需要的byte大小：
        desired_byte = math.ceil(bits / 8)
        # 输出容器的byte大小：
        output_byte = math.ceil(desired_byte / 16) * 16
        # 需要加密的元素个数
        input_num = output_byte // element_byte
        # 输出的元素个数
        output_num = output_byte // element_byte


        # 当前种子每次可以加密可以填充多少个输出元素 (16 bytes 的整数倍)：
        each_gen_num = (math.ceil(seed_byte / 16) * 16) // element_byte
        # # 如果当前种子单次加密填充的输出元素大于生成bit随机数的元素个数，则只需要单次加密
        # if each_gen_num > output_num:
        #     each_gen_num = output_num
        #
        # 构建输出容器
        out_tensor = torch.empty([self.s.shape[0],output_num], dtype=data_type, device=DEVICE)

        last_out = self.s

        # 通过生成的随机数数量进行循环判断
        generated_num = 0
        # 这里AES的密钥要保证一致，此处赋值128位的常数
        key = torch.tensor([1, 1], device=DEVICE)

        # 循环生成伪随机（每次使用上一轮生成的随机数进行二次伪随机生成）
        while generated_num < output_num:
            # 如果本轮生成的随机数数量大于剩余需要生成的随机数数量，则只需要生成剩余的随机数数量
            # 构建本轮容器
            inner_encrypt = torch.empty([self.s.shape[0], each_gen_num], dtype=data_type, device=DEVICE)
            # 对上一轮的输出进行加密
            csprng.encrypt(last_out, inner_encrypt, key, "aes128", "ecb")
            if generated_num + each_gen_num > output_num:
                each_gen_num = output_num - generated_num
                out_tensor[:, generated_num: generated_num + each_gen_num] = inner_encrypt[:, :each_gen_num]
                break
            # 将本轮计算的内容赋值给输出容器
            out_tensor[:, generated_num: generated_num + each_gen_num] = inner_encrypt
            last_out = inner_encrypt

            generated_num += each_gen_num

        # 将输出容器reshape成输出形状
        out_tensor = out_tensor.view(shape[:-1]+[-1])

        return out_tensor
