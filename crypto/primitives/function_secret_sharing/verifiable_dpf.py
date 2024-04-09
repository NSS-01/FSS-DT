"""
本文件定义了安全两方计算情况下分布式点函数的函数秘密共享
在进行分布式点函数密钥产生和求值的过程中包含了分布式点函数的相关过程
本文件中的方法定义参考E. Boyle e.t.c. Function Secret Sharing: Improvements and Extensions.2016
https://dl.acm.org/doi/10.1145/2976749.2978429
"""

import os

import torch

from common.random.prg import PRG
from config.base_configs import BIT_LEN, LAMBDA, DEVICE, PRG_TYPE, HALF_RING, data_type
from crypto.primitives.auxiliary_parameter.parameter import Parameter
from crypto.primitives.function_secret_sharing.function_secret_sharing import CW
from crypto.tensor.RingTensor import RingTensor


class VerifiableDPF(object):
    @staticmethod
    def eval(x, keys, party_id):
        """
        分布式点函数EVAL过程接口
        根据输入x，参与方在本地计算函数值，即原函数f(x)的分享值

        :param x: 输入变量值x
        :param keys: 参与方关于函数分享的密钥
        :param party_id: 参与方编号
        :return: 分布式点函数的结果
        """
        return vdpf_eval(x, keys, party_id)

    @staticmethod
    def ppq(x, keys, party_id):
        return ver_ppq_dpf(x, keys, party_id)


class VerifiableDPFKey(Parameter):

    def __init__(self):
        self.s = None
        self.cw_list = []
        self.ocw = None
        self.cs = None

    @staticmethod
    def gen(num_of_keys, alpha, beta):
        """
        分布式点函数密钥生成接口
        通过该接口实现多个密钥生成
        分布式点函数：
        f(x)=b, if x = α; f(x)=0, else

        :param num_of_keys: 需要的密钥数量
        :param alpha: 分布式比较函数的参数α
        :param beta: 分布式比较函数参数b
        :return: 各个参与方（两方）的密钥(tuple)
        """
        return vdpf_gen(num_of_keys, alpha, beta)

    def save(self, name, file_path):
        # if file not exist, create it
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        # create the dict of key
        dict = self.to('cpu').to_dic()
        file_name = os.path.join(file_path, name)
        torch.save(dict, file_name)

    @staticmethod
    def load(name, file_path):
        file_name = os.path.join(file_path, name)
        dic = torch.load(file_name)
        key = VerifiableDPFKey.dic_to_key(dic)
        return key


def vdpf_gen(num_of_keys: int, alpha: RingTensor, beta):
    """
    通过伪随机数生成器并行产生各参与方的dpf密钥
    :param num_of_keys: 所需密钥数量
    :param alpha: 分布式点函数的参数α
    :param beta: 分布式点函数的参数b
    :return: 各参与方的密钥
    """
    seed_0 = torch.randint(-HALF_RING, HALF_RING - 1, [num_of_keys, LAMBDA // BIT_LEN], dtype=data_type, device=DEVICE)
    seed_1 = torch.randint(-HALF_RING, HALF_RING - 1, [num_of_keys, LAMBDA // BIT_LEN], dtype=data_type, device=DEVICE)
    # 产生伪随机数产生器的种子

    prg = PRG(PRG_TYPE, device=DEVICE)
    prg.set_seeds(seed_0)
    s_0_0 = prg.bit_random(LAMBDA)
    prg.set_seeds(seed_1)
    s_0_1 = prg.bit_random(LAMBDA)

    k0 = VerifiableDPFKey()
    k1 = VerifiableDPFKey()

    k0.s = s_0_0
    k1.s = s_0_1

    t0 = torch.zeros(num_of_keys, 1, dtype=data_type, device=DEVICE)
    t1 = torch.ones(num_of_keys, 1, dtype=data_type, device=DEVICE)

    s_last_0 = s_0_0
    s_last_1 = s_0_1

    t_last_0 = t0
    t_last_1 = t1

    for i in range(alpha.bit_len):
        s_l_0, t_l_0, s_r_0, t_r_0 = gen_vdpf_cw(s_last_0, LAMBDA)
        s_l_1, t_l_1, s_r_1, t_r_1 = gen_vdpf_cw(s_last_1, LAMBDA)

        cond = (alpha.get_bit(alpha.bit_len - 1 - i) == 0).view(-1, 1)

        l_tensors = [s_l_0, s_l_1, t_l_0, t_l_1]
        r_tensors = [s_r_0, s_r_1, t_r_0, t_r_1]

        keep_tensors = [torch.where(cond, l, r) for l, r in zip(l_tensors, r_tensors)]
        lose_tensors = [torch.where(cond, r, l) for l, r in zip(l_tensors, r_tensors)]

        s_keep_0, s_keep_1, t_keep_0, t_keep_1 = keep_tensors
        s_lose_0, s_lose_1, t_lose_0, t_lose_1 = lose_tensors

        s_cw = s_lose_0 ^ s_lose_1

        t_l_cw = t_l_0 ^ t_l_1 ^ ~cond ^ 1
        t_r_cw = t_r_0 ^ t_r_1 ^ ~cond

        cw = CW(s_cw=s_cw, t_cw_l=t_l_cw, t_cw_r=t_r_cw, lmd=LAMBDA)

        k0.cw_list.append(cw)
        k1.cw_list.append(cw)

        t_keep_cw = torch.where(cond, t_l_cw, t_r_cw)

        s_last_0 = s_keep_0 ^ (t_last_0 * s_cw)
        s_last_1 = s_keep_1 ^ (t_last_1 * s_cw)

        t_last_0 = t_keep_0 ^ (t_last_0 * t_keep_cw)
        t_last_1 = t_keep_1 ^ (t_last_1 * t_keep_cw)

    prg.set_seeds(torch.cat((s_last_0, alpha.tensor.unsqueeze(1)), dim=1))
    pi_0 = prg.bit_random(4 * LAMBDA)
    prg.set_seeds(torch.cat((s_last_1, alpha.tensor.unsqueeze(1)), dim=1))
    pi_1 = prg.bit_random(4 * LAMBDA)

    s_0_n_add_1 = s_last_0
    s_1_n_add_1 = s_last_1

    # t_0_n_add_1 = s_0_n_add_1 & 1
    # t_1_n_add_1 = s_1_n_add_1 & 1
    cs = pi_0 ^ pi_1
    k0.cs = k1.cs = cs
    k0.ocw = k1.ocw = pow(-1, t_last_1) * (
            beta.tensor - convert_tensor(s_0_n_add_1) + convert_tensor(s_1_n_add_1))

    return k0, k1


def vdpf_eval(x: RingTensor, keys: VerifiableDPFKey, party_id):
    """
    分布式点函数EVAL过程
    根据输入x，参与方在本地计算函数值，即原函数f(x)的分享值

    :param x: 输入变量值x
    :param keys: 参与方关于函数分享的密钥
    :param party_id: 参与方编号
    :return: 分布式点函数的结果
    """
    shape = x.tensor.shape
    x = x.clone()
    x.tensor = x.tensor.view(-1, 1)

    prg = PRG(PRG_TYPE, DEVICE)

    t_last = torch.tensor([party_id], device=DEVICE)
    s_last = keys.s

    for i in range(x.bit_len):
        cw = keys.cw_list[i]

        s_cw = cw.s_cw
        t_cw_l = cw.t_cw_l
        t_cw_r = cw.t_cw_r

        s_l, t_l, s_r, t_r = gen_vdpf_cw(s_last, LAMBDA)

        s1_l = s_l ^ (s_cw * t_last)
        t1_l = t_l ^ (t_cw_l * t_last)
        s1_r = s_r ^ (s_cw * t_last)
        t1_r = t_r ^ (t_cw_r * t_last)

        x_shift_bit = x.get_bit(x.bit_len - 1 - i)

        s_last = s1_r * x_shift_bit + s1_l * (1 - x_shift_bit)
        t_last = t1_r * x_shift_bit + t1_l * (1 - x_shift_bit)

    # seed = s_last + x.tensor
    seed = torch.cat((s_last, x.tensor), dim=1)

    prg.set_seeds(seed)
    pi_ = prg.bit_random(4 * LAMBDA)
    # t_last = s_last & 1

    dpf_result = pow(-1, party_id) * (convert_tensor(s_last) + t_last * keys.ocw)
    dpf_result = dpf_result.view(shape)

    seed = keys.cs ^ (pi_ ^ (keys.cs * t_last))
    # prg.set_seeds(seed[:, 0:2])
    prg.set_seeds(seed)
    h_ = prg.bit_random(2 * LAMBDA)
    # pi = keys.cs ^ h_

    return dpf_result, h_.sum(dim=1)


def convert_tensor(tensor):
    res = tensor[:, 0].view(-1, 1)
    return res


def gen_vdpf_cw(seeds, lmd):
    prg = PRG(PRG_TYPE, device=DEVICE)
    prg.set_seeds(seeds)
    if prg.kernel == 'random':
        random_bits = prg.bit_random(2 * lmd + 2)
        s_l_res = torch.empty([prg.len, 2], dtype=torch.int64)
        t_l_res = torch.empty([prg.len], dtype=torch.int64)
        s_r_res = torch.empty([prg.len, 2], dtype=torch.int64)
        t_r_res = torch.empty([prg.len], dtype=torch.int64)

        for i, single_random_res in enumerate(random_bits):
            s_l = single_random_res >> (2 + lmd * 3) & (2 ** lmd - 1)
            s_l_low = s_l & 0xFFFFFFFFFFFFFFFF
            s_l_high = s_l >> 64 & 0xFFFFFFFFFFFFFFFF
            s_l_res[i, 0] = s_l_low >> 1
            s_l_res[i, 1] = s_l_high >> 1

            t_l = single_random_res >> (1 + lmd * 2) & 1
            t_l_res[i] = t_l

            s_r = single_random_res >> (1 + lmd) & (2 ** lmd - 1)
            s_r_low = s_r & 0xFFFFFFFFFFFFFFFF
            s_r_high = s_r >> 64 & 0xFFFFFFFFFFFFFFFF
            s_r_res[i, 0] = s_r_low >> 1
            s_r_res[i, 1] = s_r_high >> 1

            t_r = single_random_res & 1
            t_r_res[i] = t_r

        return s_l_res, t_l_res, s_r_res, t_r_res

    elif prg.kernel in ['pytorch', 'MT', 'TMT']:
        random_bits = prg.bit_random(2 * lmd + 2)
        s_l_res = random_bits[:, 0: 2]

        s_r_res = random_bits[:, 2: 4]

        t_l_res = random_bits[:, 4: 5] & 1
        t_r_res = random_bits[:, 4: 5] >> 1 & 1
        return s_l_res, t_l_res, s_r_res, t_r_res

    elif prg.kernel == 'AES':
        random_bits = prg.bit_random_with_AES_kernel(2 * lmd + 2)
        s_num = 128 // BIT_LEN
        s_l_res = random_bits[..., 0:s_num]

        s_r_res = random_bits[..., s_num: s_num + s_num]

        t_l_res = random_bits[..., s_num + s_num + 1] & 1
        t_l_res = t_l_res.unsqueeze(-1)
        t_r_res = random_bits[..., s_num + s_num + 1] >> 1 & 1
        t_r_res = t_r_res.unsqueeze(-1)
        return s_l_res, t_l_res, s_r_res, t_r_res
    else:
        raise ValueError("kernel is not supported!")


def ver_ppq_dpf(x, keys, party_id):
    # 将输入展平
    shape = x.tensor.shape
    x = x.clone()
    x.tensor = x.tensor.view(-1, 1)

    d = torch.zeros_like(x.tensor, dtype=data_type, device=DEVICE)
    psg_b = torch.zeros_like(x.tensor, dtype=data_type, device=DEVICE)
    t_last = torch.tensor([party_id], dtype=data_type, device=DEVICE)
    s_last = keys.s

    for i in range(x.bit_len):
        cw = keys.cw_list[i]

        s_cw = cw.s_cw
        t_cw_l = cw.t_cw_l
        t_cw_r = cw.t_cw_r

        s_l, t_l, s_r, t_r = gen_vdpf_cw(s_last, LAMBDA)

        s1_l = s_l ^ (s_cw * t_last)
        t1_l = t_l ^ (t_cw_l * t_last)
        s1_r = s_r ^ (s_cw * t_last)
        t1_r = t_r ^ (t_cw_r * t_last)

        x_shift_bit = x.get_bit(x.bit_len - 1 - i)

        cond = (d != x_shift_bit)
        d = x_shift_bit * cond + d * ~cond

        psg_b = (psg_b ^ t_last) * cond + psg_b * ~cond

        s_last = s1_r * x_shift_bit + s1_l * (1 - x_shift_bit)
        t_last = t1_r * x_shift_bit + t1_l * (1 - x_shift_bit)

    psg_b = (psg_b ^ t_last) * d + psg_b * (1 - d)

    prg = PRG(PRG_TYPE, DEVICE)
    # seed = s_last + x.tensor
    seed = torch.cat((s_last, x.tensor), dim=1)
    prg.set_seeds(seed)
    pi_ = prg.bit_random(4 * LAMBDA)
    seed = keys.cs ^ (pi_ ^ (keys.cs * t_last))
    prg.set_seeds(seed)
    h_ = prg.bit_random(2 * LAMBDA)
    pi = RingTensor.convert_to_ring(h_.sum(dim=1))

    return psg_b.view(shape), pi
