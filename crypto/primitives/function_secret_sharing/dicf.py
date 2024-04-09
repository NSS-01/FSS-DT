"""
本文件定义了安全两方计算情况下分布式区间函数的函数秘密共享
本文件中的方法定义参考E. Boyle e.t.c. Function Secret Sharing for Mixed-Mode and Fixed-Point Secure Computation.2021
https://link.springer.com/chapter/10.1007/978-3-030-77886-6_30
"""

import torch

from config.base_configs import PRG_TYPE, HALF_RING
from crypto.tensor.ArithmeticSharedRingTensor import ArithmeticSharedRingTensor
from crypto.primitives.auxiliary_parameter.parameter import Parameter
from crypto.primitives.function_secret_sharing.dcf import dcf_eval, dcf_gen, DCFKey
from crypto.tensor.RingTensor import RingTensor


class DICF(object):
    @staticmethod
    def eval(x_shift, keys, party_id, down_bound=torch.tensor(0), upper_bound=torch.tensor(HALF_RING - 1)):
        return dicf_eval(x_shift, keys, party_id, down_bound, upper_bound)


class DICFKey(Parameter):
    """
    分布式区间函数的秘密分享密钥
    DICF的密钥是基于DCF密钥产生的

    属性:
        dcf_key: 分布式比较函数的密钥
        r: 函数的偏移量
        z: 校验位
    """

    def __init__(self):
        self.dcf_key = DCFKey()
        self.r = None
        self.z = None
        self.size = 0

    @staticmethod
    def gen(num_of_keys, down_bound=torch.tensor(0), upper_bound=torch.tensor(HALF_RING - 1)):
        """
        分布式区间函数密钥生成接口
        通过该接口实现多个密钥生成

        :param num_of_keys: 分布式区间函数密钥个数
        :param down_bound: 分布式区间函数区间下界，默认为0
        :param upper_bound: 分布式区间函数区间上界，默认为HALF_RING-1，半环
        :return: 各个参与方（两方）的密钥
        """
        return dicf_gen(num_of_keys, down_bound, upper_bound)


def dicf_gen(num_of_keys, down_bound, upper_bound):
    """
    分布式区间函数密钥产生接口，参考只用一次DCF的方法
    :param num_of_keys: 需要的密钥个数
    :param down_bound: 区间下界
    :param upper_bound: 区间上界
    :return: 产生的密钥，输入偏移量的分享值，z的分享值
    """
    r_in = RingTensor.random([num_of_keys, 1], dtype='int')
    n_sub_1 = RingTensor.convert_to_ring(torch.tensor(-1))

    gamma = r_in + n_sub_1

    b = RingTensor.convert_to_ring(torch.tensor([1], device=r_in.device))

    # 修正参数
    q1 = (upper_bound + 1)
    ap = (down_bound + r_in.tensor)
    aq = (upper_bound + r_in.tensor)
    aq1 = (upper_bound + 1 + r_in.tensor)

    # 32位的环上 此处q1不能参与大小比较
    out = ((ap > aq) + 0) - ((ap > down_bound) + 0) + ((aq1.to(torch.int64) > q1.to(torch.int64)) + 0) + (
            (aq == n_sub_1.tensor) + 0)
    out = RingTensor.convert_to_ring(out)

    k0 = DICFKey()
    k1 = DICFKey()

    keys = dcf_gen(num_of_keys, gamma, b)

    k0.dcf_key, k1.dcf_key = keys
    k0.z, k1.z = ArithmeticSharedRingTensor.share(out.squeeze(1), 2)
    k0.r, k1.r = ArithmeticSharedRingTensor.share(r_in.squeeze(1), 2)

    return k0, k1


def dicf_eval(x_shift: RingTensor, keys: DICFKey, party_id, down_bound, upper_bound):
    """
    根据输入x计算分布式区间函数的值
    :param x_shift: 经过偏移的x的公开值
    :param keys: 参与方的密钥
    :param party_id: 参与方编号
    :param down_bound: 下界
    :param upper_bound： 上界
    :return: 计算结果的分享值
    """

    p = down_bound
    q = upper_bound
    n_1 = RingTensor.convert_to_ring(torch.tensor(-1))

    q1 = (q + 1)

    xp = (x_shift.tensor + (n_1.tensor - p))
    xq1 = (x_shift.tensor + (n_1.tensor - q1))

    xp_ring = RingTensor.convert_to_ring(xp)
    xq1_ring = RingTensor.convert_to_ring(xq1)

    s_p = dcf_eval(xp_ring, keys.dcf_key, party_id, prg_type=PRG_TYPE)
    s_q = dcf_eval(xq1_ring, keys.dcf_key, party_id, prg_type=PRG_TYPE)

    res = party_id * (((x_shift.tensor > p) + 0) - (
            (x_shift.tensor.to(torch.int64) > q1.to(torch.int64)) + 0)) - s_p + s_q + keys.z.tensor

    return RingTensor(res, dtype=x_shift.dtype, device=x_shift.device)
