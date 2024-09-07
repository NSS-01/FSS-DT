"""
基于sonic实现msb
参考： Liu e.t.c Securely Outsourcing Neural Network Inference to the Cloud with Lightweight Techniques
https://ieeexplore.ieee.org/abstract/document/9674792
"""
import math
from config.base_configs import *


class MSB(object):
    """
    定义了一个名为MSB的类，该类专门用于获取给定数字最高有效位（Most Significant Bit，most_significant_bit）。
    MSB是二进制数字中的最左侧位（在无符号整数中）。
    """

    def __init__(self):
        pass

    @staticmethod
    def get_msb(x):
        """
        提取给定数字的最高有效位(most_significant_bit)

        :param x: 输入x，对其进行获取msb操作
        :return: (int)x的最高有效位
        """
        x = x.clone()
        size = x.numel()
        x.tensor = int2bit(x, size)
        carry_bit = get_carry_bit(x, size)
        msb = carry_bit ^ x.ring_tensor.tensor[:, -1].reshape(1, size)
        return msb


def int2bit(x, size):
    """
    将整数转换为其二进制表示形式。

    :param x:输入值
    :param size:二进制表示的大小
    :return:二进制数组的表示
    """

    values = x.ring_tensor.tensor.reshape(1, size)

    arr = torch.zeros(size=(size, BIT_LEN), device=DEVICE).bool()
    for i in range(0, BIT_LEN):
        arr[:, i] = ((values >> i) & 0x01).reshape(1, size)

    return arr


def get_carry_bit(x, size):
    """
    获取进位
    使用sonic论文中的思路，采用并行化的方式加快最高位进位的获取
    引入两个参数P和G，其中P_i=a_i+b_i，G_i=a_i·b_i
    对于参与方0，a为x的二进制表示，b为0
    对于参与方1，a为0，b为x的二进制表示

    :param x: 输入值
    :param size: 数的二进制位长度
    :return: 进位位
    """

    # layer = 0
    b = torch.zeros(size=x.shape, device=DEVICE).bool()
    p_layer0 = x.tensor ^ b
    g_layer0 = get_g(x, size)

    # layer = 1
    p_temp, g_temp = get_p_and_g(p_layer0, g_layer0, x.party, BIT_LEN - 1, BIT_LEN // 2 - 1, True)
    p_layer1 = torch.zeros(size=(size, BIT_LEN // 2), device=DEVICE).bool()
    g_layer1 = torch.zeros(size=(size, BIT_LEN // 2), device=DEVICE).bool()

    p_layer1[:, 1:] = p_temp
    g_layer1[:, 1:] = g_temp
    p_layer1[:, 0] = p_layer0[:, 0]
    g_layer1[:, 0] = g_layer0[:, 0]

    p_layer = p_layer1
    g_layer = g_layer1

    layer_total = int(math.log2(BIT_LEN))

    for i in range(2, layer_total - 1):
        p_layer, g_layer = get_p_and_g(p_layer, g_layer, x.party, BIT_LEN // (2 ** i), BIT_LEN // (2 ** (i + 1)), False)

    carry_bit = get_g_last_layer(p_layer, g_layer, x.party, 2, 1)
    carry_bit = carry_bit.reshape(carry_bit.size()[0])

    return carry_bit


def get_g(x, size):
    """
    获取第0层的参数G_0

    :param x:输入值
    :param size: 输入的二进制位长
    :return: 第0层参数G_0
    """

    a, b, c = x.party.compare_key_provider.get_parameters(BIT_LEN)
    a = a.to(DEVICE)
    b = b.to(DEVICE)
    c = c.to(DEVICE)

    x_prime = torch.zeros(size=(size, BIT_LEN), device=DEVICE).bool()

    if x.party.party_id == 0:
        e = x.tensor ^ a
        f = x_prime ^ b
    else:
        e = x_prime ^ a
        f = x.tensor ^ b

    x.party.send(torch.cat((e, f), dim=0))
    get_array = x.party.receive()

    length = int(get_array.shape[0] / 2)

    e_i = get_array[:length]
    f_i = get_array[length:]

    common_e = e ^ e_i
    common_f = f ^ f_i

    return (torch.tensor(x.party.party_id, dtype=data_type).to(DEVICE) & common_f & common_e) \
        ^ (common_e & b) ^ (common_f & a) ^ c


def get_p_and_g(p, g, party, in_num, out_num, is_layer1):
    """
    根据当前层的P值和G值计算下一层的P和G

    :param p: 当前层参数P
    :param g: 当前层参数G
    :param party: 参与方
    :param in_num: 输入位的个数
    :param out_num: 输出位的个数
    :param is_layer1: 是否为第一层
    :return: 计算结果p_out,g_out
    """
    if is_layer1:
        start_bit = 1
    else:
        start_bit = 0

    p_in1 = p[:, start_bit: in_num: 2]
    p_in2 = p[:, start_bit + 1: in_num: 2]
    g_in1 = g[:, start_bit: in_num: 2]
    g_in2 = g[:, start_bit + 1: in_num: 2]

    a_p1, b_p2_p, c_p1_p2 = party.compare_key_provider.get_parameters(out_num)
    a_g1, b_p2_g, c_g1_p2 = party.compare_key_provider.get_parameters(out_num)

    e_p1 = a_p1.int() ^ p_in1
    f_p2_p = b_p2_p.int() ^ p_in2
    e_g1 = a_g1.int() ^ g_in1
    f_p2_g = b_p2_g.int() ^ p_in2

    party.send(torch.cat((e_p1, f_p2_p, e_g1, f_p2_g), dim=1))
    get_array = party.receive()

    length = int(get_array.shape[1] / 4)

    e_i = get_array[:, :length]
    f_i = get_array[:, length: length * 2]

    common_e = e_p1 ^ e_i
    common_f = f_p2_p ^ f_i

    p_out = (torch.tensor(party.party_id, dtype=data_type).to(DEVICE) & common_f & common_e) ^ (common_e & b_p2_p) ^ (
            common_f & a_p1) ^ c_p1_p2

    e_i = get_array[:, length * 2:length * 3]
    f_i = get_array[:, length * 3:]

    common_e = e_g1 ^ e_i
    common_f = f_p2_g ^ f_i

    g_out = (torch.tensor(party.party_id, dtype=data_type).to(DEVICE) & common_f & common_e) ^ (common_e & b_p2_g) ^ (
            common_f & a_g1) ^ c_g1_p2
    g_out = g_out ^ g_in2

    return p_out, g_out


def get_g_last_layer(p, g, party, in_num, out_num):
    """ 对最后一层进行操作
    处理逻辑同get_P_and_G，不过最后一层不需要计算P_out

    :param p: 输入值
    :param g: 输入值
    :param party: 参与方
    :param in_num: 输入位个数
    :param out_num: 输出位个数
    :return: 计算结果g_out
    """
    p_in2 = p[:, 1: in_num: 2]
    g_in1 = g[:, 0: in_num: 2]
    g_in2 = g[:, 1: in_num: 2]

    # g_out = g_in2 ^ (g_in1 & p_in2)
    a_g1, b_p2_g, c_g1_p2 = party.compare_key_provider.get_parameters(out_num)

    e_g1 = a_g1.int() ^ g_in1
    f_p2_g = b_p2_g.int() ^ p_in2

    party.send(torch.cat((e_g1, f_p2_g), dim=1))
    get_array = party.receive()

    out_num = int(get_array.shape[1] / 2)

    e_i = get_array[:, :out_num]
    f_i = get_array[:, out_num:]

    common_e = e_g1 ^ e_i
    common_f = f_p2_g ^ f_i

    g_out = (torch.tensor(party.party_id, dtype=data_type).to(DEVICE) & common_f & common_e) ^ (common_e & b_p2_g) ^ (
            common_f & a_g1) ^ c_g1_p2
    g_out = g_out ^ g_in2

    return g_out
