import torch
import math
from application.NN.layers.SecLayer import *
import torch.nn.functional as F

data = torch.tensor([[3.167, 1.563], [1.57, 2.89]], dtype=torch.float64)
# data = torch.tensor(-4.0)
# X = torch.tensor([[[-1.6319, -1.4728, 0.0613, 0.1803],
#                    [0.6034, 0.0317, -0.0745, 0.5937],
#                    [1.3788, -0.7867, -0.2785, 1.1716]],
#
#                   [[-0.6005, -0.6803, -0.5590, -0.6525],
#                    [-0.9157, -0.8851, -0.7001, 1.7694],
#                    [-0.5898, -0.0449, -0.4379, 0.3318]]])

X = torch.tensor([[2., 2., 7.], [9., 9., 0.]])
gelu = torch.nn.GELU()

result = gelu(data)


def erf(x):
    f = lambda t: -0.0031673043 * (t ** 7) + 0.0493278356 * (t ** 5) - 0.297453931 * (t ** 3) + 1.09952043 * t

    temp = x <= 2.5
    temp = temp * (x >= -2.5)

    x_1 = torch.where(temp, f(x), x)
    x_2 = torch.where(x_1 < -2.5, -1.0, x_1)
    x_3 = torch.where(x_2 > 2.5, 1.0, x_2)

    return x_3


def GeLU(x):
    return x / 2 * (1 + erf(x / math.sqrt(2)))


def test_GELU():
    print("torch 自带的 GELU", result)

    print("=================")

    res = GeLU(data)
    print("我写的GELU函数", res)
    print("最大差值", torch.max(res - result))


def exp(x):
    exp_iterations = 8

    result = 1 + x / (2 ** exp_iterations)

    for _ in range(exp_iterations):
        result = result.square()

    return result


def test_exp(x):
    print("=======================================")
    print("================test_exp===============")

    real_res = torch.exp(x)
    print("正确结果", real_res)

    res = exp(x)
    print("模拟结果", res)

    print("最大差值")
    print(torch.max(real_res - res))
    print("=======================================")


def inv_sqrt(x):
    """
     sqrt_nr_iters : determines the number of Newton-Raphson iterations to run.
     sqrt_nr_initial (tensor): sets the initial value for the Newton-Raphson iterations.
    :param x:
    :return:
    """

    sqrt_nr_iters = 3
    sqrt_nr_initial = None

    if sqrt_nr_initial is None:
        y = exp(torch.negative(x / 2 + 0.2)) * 2.2 + 0.2

        y = y - x / 1024
    else:
        y = sqrt_nr_initial

    for _ in range(sqrt_nr_iters):
        y = y * (3 - x * y ** 2) / 2

    return y


def test_inv_sqrt(x):
    print("=======================================")
    print("============test_inv_sqrt==============")
    real_res = 1 / torch.sqrt(x)
    print("正确结果", real_res)

    res = inv_sqrt(x)
    print("模拟结果", res)

    print("最大差值")
    print(torch.max(real_res - res))
    print("=======================================")


def test_layernorm(x):
    print("X", X)

    layernorm = torch.nn.LayerNorm(3)
    print("torch.LayerNorm:", layernorm(X))

    eps = 1e-5

    _mu = (torch.sum(x, dim=-1) / 3).unsqueeze(-1)
    res1 = x - _mu
    res2 = (x - _mu) ** 2
    res3 = torch.sum(res2, dim=-1).unsqueeze(-1)
    res4 = 1 / torch.sqrt(res3 + eps)

    res = res1 * res4 * 2

    print("_mu", _mu)
    print("res1", res1)
    print("res2", res2)
    print("res3", res3)
    print("res4", res4)
    print("res", res)

    _inv_sigma = 1 / torch.sqrt(torch.sum((x - _mu) ** 2, dim=-1).unsqueeze(-1) + eps)

    z = (x - _mu) * _inv_sigma * 2 + 0

    print(z)


def test_embedding():
    print("=======================================")
    print("============test_inv_sqrt==============")

    data = torch.randint(0, 10, size=(2, 4))
    embedding = torch.nn.Embedding(num_embeddings=10, embedding_dim=3)
    weight = embedding.weight

    print("data", data)
    print("weight", weight)

    x_onehot = F.one_hot(data, num_classes=10)

    out = x_onehot * 1.0 @ weight

    print("out", out)


if __name__ == '__main__':
    # test_GELU()
    # test_inv_sqrt(data)
    # test_exp(data)

    test_layernorm(X)
    # test_embedding()
