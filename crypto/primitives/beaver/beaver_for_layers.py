"""
根据神经网络各层的需求产生所需的beaver
支持卷据层和线性层的矩阵beaver
"""

import torch

from crypto.primitives.beaver.beaver_triples import BeaverTriples


def beaver_for_conv(x, kernel, padding, stride, num_of_triples):
    """
    卷积层的矩阵beaver三元组产生方法

    :param num_of_triples:
    :param x: 输入的x
    :param kernel: 卷积核
    :param padding: 填充
    :param stride: 步长
    :return:
    """
    shapes = conv_im2col_out_shape(x.shape, kernel.shape, stride=stride, padding=padding)
    # BeaverTriples.gen_and_save("MAT", 2, num_of_triples, shapes[0], shapes[1])  # TODO 3方
    return BeaverTriples.gen("MAT", 2, num_of_triples, shapes[0], shapes[1])


def beaver_for_linear(x, weight, num_of_triples):
    """
    线性层的矩阵beaver三元组产生方法

    :param num_of_triples:
    :param x: 输入x
    :param weight: 参与运算的权重
    :return:
    """

    weight = weight.T
    # BeaverTriples.gen_and_save("MAT", 2, num_of_triples, x.shape, weight.shape)
    return BeaverTriples.gen("MAT", 2, num_of_triples, x.shape, weight.shape)


def beaver_for_avg_pooling(x, kernel_shape, padding, stride, num_of_triples):
    """
    平均池化层的矩阵beaver三元组产生方法
    :param num_of_triples:
    :param x: 输入的x
    :param kernel_shape: 卷积核
    :param padding: 填充
    :param stride: 步长
    :return:
    """
    shapes = pooling_im2col_out_shape(x.shape, kernel_shape, stride=stride, padding=padding)
    # BeaverTriples.gen_and_save("MAT", 2, num_of_triples, shapes, torch.zeros([shapes[3], 1]).shape)
    return BeaverTriples.gen("MAT", 2, num_of_triples, shapes, torch.zeros([shapes[3], 1]).shape)


def beaver_for_adaptive_avg_pooling(x, output_shape, num_of_triples):
    """
    自适应平均池化层的beaver三元组产生方法
    :param num_of_triples:
    :param x: 输入x
    :param output_shape: 输出形状
    :return:
    """
    input_shape = torch.tensor(x.shape[2:])  # x是4维张量
    output_shape = torch.tensor(output_shape)

    stride = torch.floor(input_shape / output_shape).to(torch.int64)
    kernel_size = input_shape - (output_shape - 1) * stride

    kernel_size_list = kernel_size.tolist()
    stride_list = stride.tolist()

    beaver_for_avg_pooling(x, kernel_shape=kernel_size_list[0], padding=0, stride=stride_list[0],
                           num_of_triples=num_of_triples)


def conv_im2col_out_shape(input_shape, kernel_shape, stride=1, padding=0):
    n, c, h, w = input_shape
    f, _, k, _ = kernel_shape

    h_out = (h + 2 * padding - k) // stride + 1
    w_out = (w + 2 * padding - k) // stride + 1

    im2col_output_shape = torch.zeros([n, h_out * w_out, c * k * k]).shape
    reshaped_kernel_size = torch.zeros([1, c * k * k, f]).shape  # TODO ?
    return im2col_output_shape, reshaped_kernel_size


def pooling_im2col_out_shape(input_shape, pool_size, stride=1, padding=0):
    n, c, h, w = input_shape

    h_out = (h + 2 * padding - pool_size) // stride + 1
    w_out = (w + 2 * padding - pool_size) // stride + 1

    return torch.zeros([n, c, h_out * w_out, pool_size * pool_size]).shape
