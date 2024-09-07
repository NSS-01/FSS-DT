"""
ASS基础功能测试
"""
import time

import torch
import torch.nn.functional as F

from application.NN.NNCS import NeuralNetworkCS
from application.NN.layers.SecLayer import *
from common.utils import get_time
from config.base_configs import *
from crypto.mpc.semi_honest_party import SemiHonestCS
from crypto.tensor.ArithmeticSharedRingTensor import ArithmeticSharedRingTensor
from crypto.primitives.beaver.beaver_triples import BeaverBuffer
from crypto.tensor.RingTensor import RingTensor

server = NeuralNetworkCS(type='server')
server.inference_transformer(is_transformer=True)
server.connect(('127.0.0.1', 8089), ('127.0.0.1', 8088), ('127.0.0.1', 20000), ('127.0.0.1', 20001))

# X = torch.randint(low=-10, high=10, size=[100000], dtype=torch.int64)
# Y = torch.randint(low=-10, high=10, size=[100000], dtype=torch.int64)
# X = torch.randn((2, 3, 4))
Y = torch.randn((2, 3, 4), device=DEVICE)
# X = torch.tensor([[-3.167, 1.57], [-1.563, 2.89]])
# Y = torch.tensor([[0.077, 0.54], [2.31, 0.28]])
T = torch.tensor([[0.0000, -1.5630], [1.5700, 0.0000]], device=DEVICE)

X = torch.tensor([[[-1.6319, -1.4728, 0.0613, 0.1803],
                   [0.6034, 0.0317, -0.0745, 0.5937],
                   [1.3788, -0.7867, -0.2785, 1.1716]],

                  [[-0.6005, -0.6803, -0.5590, -0.6525],
                   [-0.9157, -0.8851, -0.7001, 1.7694],
                   [-0.5898, -0.0449, -0.4379, 0.3318]]], device=DEVICE)

x_ring = RingTensor.convert_to_ring(X)

x_0, x_1 = ArithmeticSharedRingTensor.share(x_ring, 2)
server.send(x_1)
share_x = ArithmeticSharedRingTensor(x_0, server)

y_ring = RingTensor.convert_to_ring(Y)

y_0, y_1 = ArithmeticSharedRingTensor.share(y_ring, 2)
server.send(y_1)
share_y = ArithmeticSharedRingTensor(y_0, server)

t_ring = RingTensor.convert_to_ring(T)

t_0, t_1 = ArithmeticSharedRingTensor.share(t_ring, 2)
server.send(t_1)
share_t = ArithmeticSharedRingTensor(t_0, server)

# server.beaver_provider.gen_matrix_beaver_for_parties(x.shape, y.shape)
# 保证两方通信
server.send(torch.tensor(1))


def print_original_data():
    print("===============================================")
    print("X:", X)
    print("Y:", Y)
    print("T:", T)

    print("===============================================")


def test_SecGELU(x):
    print("===============================================")
    gelu = torch.nn.GELU()
    print("torch.GELU:", gelu(X))

    secGelu = SecGELU()
    share_z = secGelu.forward(x)
    print("返回结果", share_z.restore().convert_to_real_field())

    print("===============================================")


def test_div(x: ArithmeticSharedRingTensor, y: ArithmeticSharedRingTensor):
    print("===============================================")
    print(X / Y)
    share_z = x / y
    print(share_z.restore().convert_to_real_field())
    print("================================================")


# def test_pow(t):
#     print("===============================================")
#     print("明文算法", T ** 7)
#     share_res = t ** 7
#     print(share_res.restore().convert_to_real_field())
#     print("===============================================")


# def test_div_truth(x):
#     print("===============================================")
#     print(X / 1.4142)
#     share_z = secure_div_constant(x, 1.4142)
#     print(share_z.restore().convert_to_real_field())
#     print("===============================================")


def test_exp(x):
    print("===============================================")
    print("明文数据", torch.exp(-X))
    share_z = ArithmeticSharedRingTensor.nexp(x)
    print(share_z.restore().convert_to_real_field())
    print("===============================================")

    print("明文数据先exp再求和", torch.sum(torch.exp(-X), dim=-1))
    share_z = ArithmeticSharedRingTensor.nexp(x)
    share_z = share_z.sum(dim=-1)
    print("密文求和", share_z.restore().convert_to_real_field())
    print("===============================================")


def test_inv_sqrt(x):
    print("===============================================")
    print("明文数据", 1 / torch.sqrt(X))
    share_z = ArithmeticSharedRingTensor.reciprocal_sqrt(x)
    print(share_z.restore().convert_to_real_field())
    print(torch.max((1 / torch.sqrt(X)) - share_z.restore().convert_to_real_field()))
    print("===============================================")


def test_Softmax(x):
    print("===============================================")
    softmax = torch.nn.Softmax(dim=-1)
    print("torch.softmax:", softmax(X))

    secSoftmax = SecSoftmax(dim=-1)
    share_z = secSoftmax.forward(x)
    print("返回结果", share_z.restore().convert_to_real_field())

    print("===============================================")


def test_LayerNorm(x):
    print("===============================================")
    layernorm = torch.nn.LayerNorm(4)
    print("torch.LayerNorm:", layernorm(X))

    secLayerNorm = SecLayerNorm(4)
    share_z = secLayerNorm.forward(x)

    print("返回结果", share_z.restore().convert_to_real_field())

    print("===============================================")


def test_embedding():
    print("===============================================")

    data = torch.randint(0, 10, size=(2, 3), dtype=torch.int64)
    data_onehot = F.one_hot(data, num_classes=10) * 1.0

    x_ring = RingTensor.convert_to_ring(data_onehot).to(DEVICE)

    x_0, x_1 = ArithmeticSharedRingTensor.share(x_ring, 2)
    server.send(x_1)
    share_x = ArithmeticSharedRingTensor(x_0, server)

    embedding = torch.nn.Embedding(10, 4)
    print("torch.Embedding:", embedding(data))

    weight = embedding.weight.to(DEVICE)

    weight = RingTensor.convert_to_ring(weight)
    weight_0, weight_1 = ArithmeticSharedRingTensor.share(weight, 2)
    server.send(weight_1)

    secembedding = SecEmbedding(10, 4, _weight=weight_0)
    share_z = secembedding.forward(share_x)

    print("返回结果", share_z.restore().convert_to_real_field())

    print("===============================================")


if __name__ == '__main__':
    print_original_data()
    # get_time(test_div, share_x, share_y)
    # # test_div_truth(share_x)
    # get_time(test_SecGELU, share_x)
    # # test_pow(share_t)
    # get_time(test_exp, share_x)
    # get_time(test_inv_sqrt, share_x)
    # get_time(test_Softmax, share_x)
    get_time(test_LayerNorm, share_x)
    get_time(test_embedding)
