import torch

from config.base_configs import GE_TYPE, DEVICE, data_type
from crypto.protocols.most_significant_bit.most_significant_bit import MSB
from crypto.tensor.RingTensor import RingTensor


def beaver_mul(x, y):
    """
    利用beaver三元组进行ASS乘法

    :param x:  参与乘法的ASS x
    :param y:  参与乘法的ASS y
    :return: res: 乘法运算结果
    """
    a, b, c = x.party.beaver_provider.get_triples(x.shape)  # type是ASS
    a.dtype = b.dtype = c.dtype = x.dtype

    e = x - a
    f = y - b

    common_e = e.restore()
    common_f = f.restore()

    i = x.party.party_id

    res1 = RingTensor.mul(common_e, common_f) * i
    res2 = RingTensor.mul(a, common_f)
    res3 = RingTensor.mul(common_e, b)
    res = res1 + res2 + res3 + c

    res = x.__class__(res, x.party)

    return res


def secure_matmul(x, y):
    """
    利用beaver三元组进行ASS矩阵乘法

    :param x: 参与乘法的ASS矩阵x
    :param y: 参与乘法的ASS矩阵y
    :return: res: 矩阵运算结果
    """
    a_matrix, b_matrix, c_matrix = x.party.beaver_provider.get_mat_beaver(x.shape, y.shape)

    e = x - a_matrix
    f = y - b_matrix

    common_e = e.restore()
    common_f = f.restore()

    res1 = RingTensor.matmul(common_e, common_f)
    res2 = RingTensor.matmul(common_e, b_matrix)
    res3 = RingTensor.matmul(a_matrix, common_f)

    res = res1 * x.party.party_id + res2 + res3 + c_matrix

    res = x.__class__(res, x.party)

    return res


def secure_ge(x, y):
    """
    ASS的大小比较

    :param x:
    :param y:
    :return:
    """
    ge_methods = {'MSB': msb_ge, 'FSS': fss_ge, 'GROTTO': grotto_ge, 'SIGMA': sigma_ge}
    return ge_methods[GE_TYPE](x, y)


def msb_ge(x, y):
    z = x - y
    shape = z.shape
    msb = MSB.get_msb(z)
    # msb为0，z>=0；msb为1，z<0。而True为1，False为0，所以在输出时应反转一下0和1
    ge_res = msb
    if x.party.party_id == 0:
        ge_res = msb ^ torch.ones_like(msb, dtype=data_type)
    ge_res = b2a(ge_res, x.party)
    ge_res = ge_res.reshape(shape)
    ge_res.dtype = x.dtype
    ge_res = ge_res * x.scale
    return ge_res


def fss_ge(x, y):
    z = x - y
    shape = z.shape
    key = x.party.compare_key_provider.get_parameters(x.numel())
    z_shift = x.__class__(key.r, x.party) + z.flatten()
    z_shift = z_shift.restore()
    from crypto.primitives.function_secret_sharing.dicf import DICF
    ge_res = DICF.eval(z_shift, key, x.party.party_id).view(shape) * x.scale
    ge_res.dtype = x.dtype
    return x.__class__(ge_res, x.party)


def grotto_ge(x, y):
    z = x - y
    shape = z.shape
    key = x.party.compare_key_provider.get_parameters(x.numel())
    z_shift = x.__class__(key.phi, x.party) - z.flatten()
    z_shift = z_shift.restore()
    from crypto.primitives.function_secret_sharing.p_dicf import ParityDICF
    ge_res = ParityDICF.eval(z_shift, key, x.party.party_id).view(shape)
    ge_res = b2a(ge_res, x.party)
    ge_res.dtype = x.dtype
    return ge_res * x.scale


def sigma_ge(x, y):
    z = x - y
    shape = z.shape
    key = x.party.compare_key_provider.get_parameters(x.numel())
    z_shift = x.__class__(key.r_in, x.party) + z.flatten()
    z_shift = z_shift.restore()
    from crypto.protocols.comparison.cmp_sigma import CMPSigma
    ge_res = CMPSigma.eval(z_shift, key, x.party.party_id).view(shape)
    ge_res = b2a(ge_res.tensor, x.party)
    ge_res.dtype = x.dtype
    return ge_res * x.scale


def b2a(x: torch.Tensor, party):
    """
    基于sonic的secure B2A

    :param x: 布尔秘密共享
    :param party: 参与方id
    :return: 转换后的算术秘密共享
    """
    from crypto.tensor.ArithmeticSharedRingTensor import ArithmeticSharedRingTensor
    zero = RingTensor.zeros(x.shape, device=DEVICE)
    x = RingTensor(x, 'int')
    if party.party_id == 0:
        a = ArithmeticSharedRingTensor(x, party)
        b = ArithmeticSharedRingTensor(zero, party)
    else:
        b = ArithmeticSharedRingTensor(x, party)
        a = ArithmeticSharedRingTensor(zero, party)
    return a + b - a * b * 2


def truncate(share, scale=None):
    from crypto.protocols.truncate import crypten
    if scale is None:
        scale = share.scale
    return crypten.truncate(share, scale)


def secure_neg_exp(x, scale_bit=16):
    """
    exp(-x)的安全计算
    :param x: 输入x为正数，x的有效范围是[0, 16)
    :param scale_bit:
    :return:
    """
    from crypto.protocols.exp.neg_exp import NegExp
    key = x.party.neg_exp_provider.get_parameters(x.numel())
    return NegExp.eval(x, key, scale_bit)


def secure_pos_exp(x, scale_bit=16):
    """
    exp(x)的安全计算
    :param x: x的有效范围是[0, 21.5) 上界在21.4和21.5之间
    :param scale_bit:
    :return:
    """
    from crypto.protocols.exp.pos_exp import PosExp
    key = x.party.pos_exp_provider.get_parameters(x.numel())
    return PosExp.eval(x, key, scale_bit)


def secure_exp(x, scale_bit=16):
    """
    计算exp(x)
    :param x: x的有效范围是(-16, 21.5) 上界在21.4和21.5之间
    :param scale_bit:
    :return:
    """

    # is_pos = x >= x.__class__(RingTensor.zeros_like(x), party=x.party)
    #
    # a = is_pos * x * 2 - x  # a = abs(x)
    #
    # return is_pos * secure_pos_exp(a, scale_bit) + (-is_pos + RingTensor.ones_like(is_pos)) * secure_neg_exp(a,
    #                                                                                                          scale_bit)
    from crypto.protocols.exp.exp import Exp
    key = x.party.exp_provider.get_parameters(x.numel())
    return Exp.eval(x, key, scale_bit)


def secure_exp_by_lim(x):
    """
    利用极限近似计算exp

    Approximates the exponential function using a limit approximation:

    .. math::

        exp(x) = \lim_{n \\rightarrow \\infty} (1 + x / n) ^ n

    Here we compute exp by choosing n = 2 ** d for some large d equal to
    `iterations`. We then compute (1 + x / n) once and square `d` times.

    Set the number of iterations for the limit approximation with
    config.exp_iterations.
    :param x:
    :return:
    """

    exp_iterations = 8

    result = x / (2 ** exp_iterations) + RingTensor.ones_like(x)

    for _ in range(exp_iterations):
        result = result * result

    return result


def secure_div(x, y):
    """
    牛顿迭代法实现ASS的除法协议
    TODO： 仅在y > 0时正确，仅支持64位

    :param x: 被除数
    :param y: 除数
    :return: 商
    """

    powers = [(2 ** (i - 1)) * 1.0 for i in range(-15, y.ring_tensor.bit_len - 48)]
    powers = RingTensor.convert_to_ring(powers).to(x.device)

    for _ in range(len(y.shape)):
        powers = powers.unsqueeze(-1)

    powers = x.__class__(powers, party=x.party)

    k = (y.unsqueeze(0) > powers).sum(dim=0)

    clear_k = k.restore().convert_to_real_field() - 15

    k_range = 2 ** (clear_k + 1)

    ring_k_range = RingTensor.convert_to_ring(k_range)

    a = x / ring_k_range
    b = y / ring_k_range

    w = b * (-2) + RingTensor.full_like(x, 2.9142)

    e0 = -(b * w) + RingTensor.ones_like(x)

    e1 = e0 * e0
    res = a * w * (e0 + RingTensor.ones_like(x)) * (e1 + RingTensor.ones_like(x))

    return res


def secure_reciprocal_sqrt(x):
    """
    使用 Newton-Raphson 方法计算输入的倒数平方根
    Computes the inverse square root of the input using the Newton-Raphson method.

    sqrt_nr_iters : determines the number of Newton-Raphson iterations to run.
    sqrt_nr_initial: sets the initial value for the Newton-Raphson iterations.
    :return:
    """

    sqrt_nr_iters = 3
    sqrt_nr_initial = None

    zero_share = x.__class__(RingTensor.zeros_like(x), party=x.party)

    a = RingTensor.full_like(x, fill_value=0.2)
    b = RingTensor.full_like(x, fill_value=2.2)

    c = RingTensor.full_like(x.ring_tensor, fill_value=3.0)

    if sqrt_nr_initial is None:
        y = x.__class__.nexp(x / 2 + a) * b + a
        y = y - x / 1024
    else:
        y = sqrt_nr_initial

    for _ in range(sqrt_nr_iters):
        y = y * (zero_share - x * y * y + c)
        y = y / 2

    return y
