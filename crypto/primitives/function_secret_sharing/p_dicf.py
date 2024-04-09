"""
本文件中定义通过(2, 2)-DPFs实现Z2𝓃的超快速(2 + 1)-PC方法
主要参考论文：Storrier K, Vadapalli A, Lyons A, et al. Grotto: Screaming fast (2+ 1)-PC for ℤ2n via (2, 2)-DPFs[J].
IACR Cryptol. ePrint Arch., 2023, 2023: 108.
"""

import torch

from common.random.prg import PRG
from config.base_configs import LAMBDA, DEVICE, data_type, PRG_TYPE, HALF_RING
from crypto.tensor.ArithmeticSharedRingTensor import ArithmeticSharedRingTensor
from crypto.primitives.auxiliary_parameter.parameter import Parameter
from crypto.primitives.function_secret_sharing.dpf import DPFKey, dpf_gen, gen_dpf_cw
from crypto.tensor.RingTensor import RingTensor


class ParityDICF(object):
    @staticmethod
    def eval(x_shift: RingTensor, key, party_id, prg_type=PRG_TYPE, down_bound=torch.tensor(0),
             upper_bound=torch.tensor(HALF_RING - 1)):
        return parity_dicf_eval(x_shift, key, party_id, prg_type, down_bound, upper_bound)


class ParityDICFKey(Parameter):
    def __init__(self):
        self.dpf_key = DPFKey()
        self.phi = None

    @staticmethod
    def gen(num_of_keys, beta=RingTensor.convert_to_ring(torch.tensor(1))):
        return parity_dicf_gen(num_of_keys, beta)


def parity_dicf_gen(num_of_keys: int, beta):
    """
    通过伪随机数生成器并行产生各参与方的dpf密钥
    :param num_of_keys: 所需密钥数量
    :param beta: 分布式点函数的参数b
    :return: 各参与方的密钥
    """
    phi = RingTensor.random([num_of_keys])
    k0, k1 = ParityDICFKey(), ParityDICFKey()
    k0.dpf_key, k1.dpf_key = dpf_gen(num_of_keys, phi, beta)
    k0.phi, k1.phi = ArithmeticSharedRingTensor.share(phi, 2)
    return k0, k1


def parity_dicf_eval(x_shift: RingTensor, key: ParityDICFKey, party_id, prg_type, down_bound, upper_bound):
    """
    根据GROOTO论文中的方法，实现通过一次DPF实现DICF的过程
    :param party_id:
    :param x_shift:经过偏移的x的公开值
    :param key: 参与方关于函数分享的密钥
    :param prg_type: 随机数种子生成器类型
    :param down_bound: 区间上界
    :param upper_bound: 区间下界
    :return: 某一方根据输入值的判断是否在区间上
    """
    p = RingTensor.convert_to_ring(down_bound + x_shift.tensor)
    q = RingTensor.convert_to_ring(upper_bound + x_shift.tensor)

    cond = (p.tensor ^ q.tensor) < 0
    tau = ((p > q) ^ cond) * party_id

    x = torch.stack([p.tensor, q.tensor]).view(2, -1, 1)
    x = RingTensor(x, dtype=x_shift.dtype, device=DEVICE)

    parity_x = pps_calculate(x, key.dpf_key, party_id, prg_type)
    parity_p = parity_x[0].view(x_shift.shape)
    parity_q = parity_x[1].view(x_shift.shape)

    ans = parity_p ^ parity_q ^ tau

    return ans


def pps_calculate(x: RingTensor, keys: DPFKey, party_id, prg_type=PRG_TYPE):
    """
    分布式点函数EVAL过程改造成的计算某一部分的前缀奇偶校验和(Prefix Parity Sum)
    根据输入x，参与方在本地计算该点在构造树上的奇偶校验性
    :param x:输入变量值x
    :param keys: 参与方关于函数分享的密钥
    :param party_id: 参与方编号
    :param prg_type: 伪随机数产生器的类型
    :return: 前缀奇偶校验的结果
    """
    prg = PRG(prg_type, DEVICE)

    x = x.clone()

    d = torch.zeros_like(x.tensor, dtype=data_type, device=DEVICE)
    psg_b = torch.zeros_like(x.tensor, dtype=data_type, device=DEVICE)
    t_last = torch.tensor([party_id], dtype=data_type, device=DEVICE)
    s_last = keys.s
    for i in range(x.bit_len):
        cw = keys.cw_list[i]

        s_cw = cw.s_cw
        t_cw_l = cw.t_cw_l
        t_cw_r = cw.t_cw_r

        s_l, t_l, s_r, t_r = gen_dpf_cw(prg, s_last, LAMBDA)

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

    return psg_b
