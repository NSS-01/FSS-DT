"""
本文件定义了安全两方计算情况下分布式比较函数的函数秘密共享
本文件中的方法定义参考E. Boyle e.t.c. Function Secret Sharing for Mixed-Mode and Fixed-Point Secure Computation.2021
https://link.springer.com/chapter/10.1007/978-3-030-77886-6_30
"""

import time

import torch

from common.random.prg import PRG
from config.base_configs import BIT_LEN, LAMBDA, DEVICE, PRG_TYPE, data_type, HALF_RING
from crypto.primitives.auxiliary_parameter.parameter import Parameter
from crypto.primitives.function_secret_sharing.function_secret_sharing import CW, CWList
from crypto.tensor.RingTensor import RingTensor


class DCF(object):
    @staticmethod
    def eval(x, keys, party_id, prg_type=PRG_TYPE):
        """
        分布式比较函数EVAL过程接口
        根据输入x，参与方在本地计算函数值，即原函数f(x)的分享值

        :param x: 输入变量值x
        :param keys: 参与方关于函数分享的密钥
        :param party_id: 参与方编号
        :param prg_type: 伪随机数产生器的类型
        :return: 分布式比较函数的结果
        """
        return dcf_eval(x, keys, party_id, prg_type)


class DCFKey(Parameter):
    """
    分布式比较函数的秘密分享密钥

    属性:
        s: DCF产生的树根节点的参数，λ位01串
        cw_list: 校验字列表
        ex_cw_dcf: 额外校验字用于dcf求值
    """

    def __init__(self):
        self.s = None
        self.cw_list = CWList()
        self.ex_cw_dcf = None
        self.size = 0

    @staticmethod
    def gen(num_of_keys, alpha, beta):
        """
        分布式比较函数密钥生成接口
        通过该接口实现多个密钥生成
        分布式比较函数：
            f(x)=b, if x < α; f(x)=0, else

        :param num_of_keys: 需要的密钥个数
        :param alpha: 分布式比较函数的参数α
        :param beta: 分布式比较函数参数b
        :return: 各个参与方（两方）的密钥
        """
        return dcf_gen(num_of_keys, alpha, beta)


def dcf_gen(num_of_keys, alpha: RingTensor, beta):
    """
    通过伪随机数生成器产生各参与方的dcf密钥
    :param num_of_keys: 需要的密钥数量
    :param alpha: 分布式比较函数的参数α
    :param beta: 分布式比较函数的参数b
    :return: 参与双方的密钥(tuple)
    """

    # 产生伪随机数产生器的种子
    seed_0 = torch.randint(-HALF_RING, HALF_RING - 1, [num_of_keys, LAMBDA // BIT_LEN], dtype=data_type, device=DEVICE)
    seed_1 = torch.randint(-HALF_RING, HALF_RING - 1, [num_of_keys, LAMBDA // BIT_LEN], dtype=data_type, device=DEVICE)

    prg = PRG(PRG_TYPE, device=DEVICE)
    prg.set_seeds(seed_0)
    s_0_0 = prg.bit_random(LAMBDA)
    prg.set_seeds(seed_1)
    s_0_1 = prg.bit_random(LAMBDA)

    k0 = DCFKey()
    k1 = DCFKey()

    k0.s = s_0_0
    k1.s = s_0_1

    t0 = torch.zeros(num_of_keys, 1, dtype=data_type, device=DEVICE)
    t1 = torch.ones(num_of_keys, 1, dtype=data_type, device=DEVICE)

    s_last_0 = s_0_0
    s_last_1 = s_0_1

    t_last_0 = t0
    t_last_1 = t1

    v_a = torch.zeros((num_of_keys, 1), dtype=data_type, device=DEVICE)

    for i in range(alpha.bit_len):
        s_l_0, v_l_0, t_l_0, s_r_0, v_r_0, t_r_0 = gen_dcf_cw(prg, s_last_0, LAMBDA)
        s_l_1, v_l_1, t_l_1, s_r_1, v_r_1, t_r_1 = gen_dcf_cw(prg, s_last_1, LAMBDA)

        cond = (alpha.get_bit(alpha.bit_len - 1 - i) == 0).view(-1, 1)

        l_tensors = [s_l_0, s_l_1, v_l_0, v_l_1, t_l_0, t_l_1]
        r_tensors = [s_r_0, s_r_1, v_r_0, v_r_1, t_r_0, t_r_1]

        keep_tensors = [torch.where(cond, l, r) for l, r in zip(l_tensors, r_tensors)]
        lose_tensors = [torch.where(cond, r, l) for l, r in zip(l_tensors, r_tensors)]

        s_keep_0, s_keep_1, v_keep_0, v_keep_1, t_keep_0, t_keep_1 = keep_tensors
        s_lose_0, s_lose_1, v_lose_0, v_lose_1, t_lose_0, t_lose_1 = lose_tensors

        s_cw = s_lose_0 ^ s_lose_1

        v_cw = torch.pow(-1, t_last_1) * (
                convert_tensor(v_lose_1)
                - convert_tensor(v_lose_0)
                - v_a)

        v_cw = torch.where(alpha.get_bit(alpha.bit_len - 1 - i) == 1, v_cw + pow(-1, t_last_1) * beta.tensor, v_cw)

        v_a = (v_a
               - convert_tensor(v_keep_1)
               + convert_tensor(v_keep_0)
               + pow(-1, t_last_1) * v_cw)

        t_l_cw = t_l_0 ^ t_l_1 ^ ~cond ^ 1
        t_r_cw = t_r_0 ^ t_r_1 ^ ~cond

        cw = CW(s_cw=s_cw, v_cw=v_cw, t_cw_l=t_l_cw, t_cw_r=t_r_cw, lmd=LAMBDA)

        k0.cw_list.append(cw)
        k1.cw_list.append(cw)

        t_keep_cw = torch.where(cond, t_l_cw, t_r_cw)

        s_last_0 = s_keep_0 ^ (t_last_0 * s_cw)
        s_last_1 = s_keep_1 ^ (t_last_1 * s_cw)

        t_last_0 = t_keep_0 ^ (t_last_0 * t_keep_cw)
        t_last_1 = t_keep_1 ^ (t_last_1 * t_keep_cw)

    k0.ex_cw_dcf = k1.ex_cw_dcf = pow(-1, t_last_1) * (
            convert_tensor(s_last_1)
            - convert_tensor(s_last_0)
            - v_a[:, 0].view(-1, 1))

    return k0, k1


def dcf_eval(x: RingTensor, keys: DCFKey, party_id, prg_type):
    """
    分布式比较函数EVAL过程
    根据输入x，参与方在本地计算函数值，即原函数f(x)的分享值

    :param x: 输入变量值x
    :param keys: 参与方关于函数分享的密钥
    :param party_id: 参与方编号
    :param prg_type: 伪随机数产生器的类型
    :return: 分布式比较函数的结果
    """
    # flatten the input tensor and reshape it back after computation
    shape = x.tensor.shape
    x = x.clone()
    x.tensor = x.tensor.view(-1, 1)

    prg = PRG(prg_type, DEVICE)
    t_last = torch.tensor([party_id], dtype=data_type, device=DEVICE)
    dcf_result = torch.zeros_like(x.tensor)
    s_last = keys.s
    key_time = 0

    for i in range(x.bit_len):
        cw = keys.cw_list[i]

        s_cw = cw.s_cw
        v_cw = cw.v_cw
        t_cw_l = cw.t_cw_l
        t_cw_r = cw.t_cw_r

        k_s = time.time()
        s_l, v_l, t_l, s_r, v_r, t_r = gen_dcf_cw(prg, s_last, LAMBDA)
        k_e = time.time()
        key_time += k_e - k_s

        s1_l = s_l ^ (s_cw * t_last)
        t1_l = t_l ^ (t_cw_l * t_last)
        s1_r = s_r ^ (s_cw * t_last)
        t1_r = t_r ^ (t_cw_r * t_last)

        x_shift_bit = x.get_bit(x.bit_len - 1 - i)

        v_curr = v_r * x_shift_bit + v_l * (1 - x_shift_bit)
        dcf_result = dcf_result + pow(-1, party_id) * (convert_tensor(v_curr) + t_last * v_cw)

        s_last = s1_r * x_shift_bit + s1_l * (1 - x_shift_bit)
        t_last = t1_r * x_shift_bit + t1_l * (1 - x_shift_bit)

    dcf_result = dcf_result + pow(-1, party_id) * (
            convert_tensor(s_last) + t_last * keys.ex_cw_dcf)

    return dcf_result.view(shape)


def gen_dcf_cw(prg, new_seeds, lmd):
    prg.set_seeds(new_seeds)
    if prg.kernel == 'random':
        random_bits = prg.bit_random(4 * lmd + 2)
        s_l_res = torch.empty([prg.len, 2], dtype=data_type)
        v_l_res = torch.empty([prg.len, 2], dtype=data_type)
        t_l_res = torch.empty([prg.len], dtype=data_type)
        s_r_res = torch.empty([prg.len, 2], dtype=data_type)
        v_r_res = torch.empty([prg.len, 2], dtype=data_type)
        t_r_res = torch.empty([prg.len], dtype=data_type)

        for i, single_random_res in enumerate(random_bits):
            s_l = single_random_res >> (2 + lmd * 3) & (2 ** lmd - 1)
            s_l_low = s_l & 0xFFFFFFFFFFFFFFFF
            s_l_high = s_l >> 64 & 0xFFFFFFFFFFFFFFFF
            s_l_res[i, 0] = s_l_low >> 1
            s_l_res[i, 1] = s_l_high >> 1

            v_l = single_random_res >> (2 + lmd * 2) & (2 ** lmd - 1)
            v_l_low = v_l & 0xFFFFFFFFFFFFFFFF
            v_l_high = v_l >> 64 & 0xFFFFFFFFFFFFFFFF
            v_l_res[i, 0] = v_l_low >> 1
            v_l_res[i, 1] = v_l_high >> 1

            t_l = single_random_res >> (1 + lmd * 2) & 1
            t_l_res[i] = t_l

            s_r = single_random_res >> (1 + lmd) & (2 ** lmd - 1)
            s_r_low = s_r & 0xFFFFFFFFFFFFFFFF
            s_r_high = s_r >> 64 & 0xFFFFFFFFFFFFFFFF
            s_r_res[i, 0] = s_r_low >> 1
            s_r_res[i, 1] = s_r_high >> 1

            v_r = single_random_res >> 1 & (2 ** lmd - 1)
            v_r_low = v_r & 0xFFFFFFFFFFFFFFFF
            v_r_high = v_r >> 64 & 0xFFFFFFFFFFFFFFFF
            v_r_res[i, 0] = v_r_low >> 1
            v_r_res[i, 1] = v_r_high >> 1

            t_r = single_random_res & 1
            t_r_res[i] = t_r

        return s_l_res, v_l_res, t_l_res, s_r_res, v_r_res, t_r_res

    elif prg.kernel in ['pytorch', 'MT', 'TMT']:
        random_bits = prg.bit_random(4 * lmd + 2)
        s_l_res = random_bits[:, 0: 2]
        v_l_res = random_bits[:, 2: 4]

        s_r_res = random_bits[:, 4: 6]
        v_r_res = random_bits[:, 6: 8]

        t_l_res = random_bits[:, 8: 9] & 1
        t_r_res = random_bits[:, 8: 9] >> 1 & 1
        return s_l_res, v_l_res, t_l_res, s_r_res, v_r_res, t_r_res
    elif prg.kernel == 'AES':
        random_bits = prg.bit_random_with_AES_kernel(4 * lmd + 2)
        s_num = 128 // BIT_LEN
        s_l_res = random_bits[..., 0:s_num]
        v_l_res = random_bits[..., s_num: s_num + s_num]

        s_r_res = random_bits[..., 2 * s_num: 2 * s_num + s_num]

        v_r_res = random_bits[..., 3 * s_num: 3 * s_num + s_num]

        t_l_res = random_bits[..., 4 * s_num + 1] & 1
        t_l_res = t_l_res.unsqueeze(-1)
        t_r_res = random_bits[..., 4 * s_num + 1] >> 1 & 1
        t_r_res = t_r_res.unsqueeze(-1)
        return s_l_res, v_l_res, t_l_res, s_r_res, v_r_res, t_r_res
    else:
        raise ValueError("kernel is not supported!")


def convert_tensor(tensor):
    res = tensor[:, 0].view(-1, 1)
    return res
