import torch
import math
from config.base_configs import data_type, RING_MAX


def cuda_matmul(x, y):
    if data_type is torch.int32:
        return cuda_matmul_32(x, y)
    else:  # data_type is torch.int64
        return cuda_matmul_64(x, y)


def cuda_matmul_32(x, y):
    """
    在CUDA上执行两个int32整数矩阵的乘法。
    由于CUDA不直接支持整数矩阵的乘法，考虑直接转为64位浮点数进行运算精度受到限制，因此对于int32类型的矩阵，
    此函数首先将输入矩阵的每个元素分为高位(8bit)和低位（24bit）部分。然后，使用这些高低位元素进行四次浮点矩阵乘法，
    最后将这四个结果加在一起以得到最终的整数矩阵乘法结果。

    :param x:(torch.Tensor)第一个整数矩阵
    :param y:(torch.Tensor)第二个整数矩阵
    :return:(torch.Tensor)x@y的结果，返回值为整数矩阵
    """
    tag = 2 ** 24

    x_high = torch.floor(x / tag).to(torch.float64)
    x_low = (x - x_high * tag).to(torch.float64)

    y_high = torch.floor(y / tag).to(torch.float64)
    y_low = (y - y_high * tag).to(torch.float64)

    result = (torch.matmul(x_high, y_high) * tag * tag % RING_MAX +
              torch.matmul(x_high, y_low) * tag % RING_MAX +
              torch.matmul(x_low, y_high) * tag % RING_MAX +
              torch.matmul(x_low, y_low)) % RING_MAX

    return result.to(torch.int64).to(torch.int32)  # 需要先转成int64再转成int32, 否则出错


def cuda_matmul_64(x, y):
    """
    在CUDA上执行两个int64整数矩阵的乘法。
    由于CUDA不直接支持整数矩阵的乘法，考虑直接转为64位浮点数进行运算精度受到限制，因此对于int64类型的矩阵，
    将每个矩阵拆分成4个块，每块16位，分别运算再相加得到最终结果。
    参考CrypTen

    :param x:(torch.Tensor)第一个整数矩阵
    :param y:(torch.Tensor)第二个整数矩阵
    :return:(torch.Tensor)x@y结果，返回值为整数矩阵
    """
    block_num = 4

    # 拆分成4份
    x_block = split_matrix(x, block_num)
    y_block = split_matrix(y, block_num)

    result = 0
    # TODO: 可不可以改成并行化
    for i in range(block_num):
        for j in range(block_num):
            if (i + j) * block_num >= 64:  # BIT_LEN == 64
                continue
            shift = (i + j) * 16  # BIT_LEN / block_num
            result += torch.matmul(x_block[i], y_block[j]).long() << shift

    return result


def split_matrix(x, block_num, bit_len=64):
    """
    将矩阵按照tag进行拆分
    :param x: 要拆分的矩阵
    :param block_num: 拆分的块数
    :param bit_len: 原位长
    :return:
    """
    block_size = math.ceil(bit_len / block_num)
    tag = 2 ** block_size

    x_ = x
    x_block = []

    for _ in range(block_num - 1):
        x_block.append((x_ % tag).to(torch.float64))
        x_ = x_ >> block_size
    x_block.append(x_.to(torch.float64))
    return x_block
