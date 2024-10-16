import torch

from NssMPC import RingTensor, ArithmeticSecretSharing
from application.neural_network.layers import SecGeLU, SecSoftmax, SecLayerNorm, SecTanh
from application.neural_network.party import NeuralNetworkCS
from NssMPC.common.utils import get_time, comm_count
from NssMPC.config import DEVICE

server = NeuralNetworkCS(type='server')
server.activate_by_gelu()
server.online()

# X = torch.randint(low=-10, high=10, size=[100000], dtype=torch.int64)
# Y = torch.randint(low=-10, high=10, size=[100000], dtype=torch.int64)
# X = torch.randn((2, 3, 4))
# Y = torch.rand((2, 3, 4), device=DEVICE)
# X = torch.tensor([[-3.167, 1.57], [-1.563, 2.89]])
# Y = torch.tensor([[0.077, 0.54], [2.31, 0.28]])
# T = torch.tensor([[0.0000, -1.5630], [1.5700, 0.0000]], device=DEVICE)
#
# X = torch.tensor([[[-1.6319, -1.4728, 0.0613, 0.1803],
#                    [0.6034, 0.0317, -0.0745, 0.5937],
#                    [1.3788, -0.7867, -0.2785, 1.1716]],
#
#                   [[-0.6005, -0.6803, -0.5590, -0.6525],
#                    [-0.9157, -0.8851, -0.7001, 1.7694],
#                    [-0.5898, -0.0449, -0.4379, 0.3318]]], device=DEVICE)
#
# Y = torch.tensor([[[1.6319, 1.4728, 0.0613, 0.1803],
#                    [0.6034, 0.0317, 0.0745, 0.5937],
#                    [1.3788, 0.7867, 0.2785, 1.1716]],
#
#                   [[0.6005, 0.6803, 0.5590, 0.6525],
#                    [0.9157, 0.8851, 0.7001, 1.7694],
#                    [0.5898, 0.0449, 0.4379, 0.3318]]], device=DEVICE)
# X = torch.rand([1, 35, 768], device=DEVICE)
# Y = torch.rand([1, 12, 35, 35], device=DEVICE)
X = torch.rand([1, 100, 100], device=DEVICE)
Y = torch.rand([1, 1, 100, 100], device=DEVICE)
T = torch.rand([1, 35, 35], device=DEVICE)

x_ring = RingTensor.convert_to_ring(X)

x_0, x_1 = ArithmeticSecretSharing.share(x_ring, 2)
server.send(x_1)
share_x = x_0
share_x.party = server

y_ring = RingTensor.convert_to_ring(Y)

y_0, y_1 = ArithmeticSecretSharing.share(y_ring, 2)
server.send(y_1)
share_y = y_0
share_y.party = server

t_ring = RingTensor.convert_to_ring(T)

t_0, t_1 = ArithmeticSecretSharing.share(t_ring, 2)
server.send(t_1)
share_t = t_0
share_t.party = server

# server.beaver_provider.gen_matrix_beaver_for_parties(x.shape, y.shape)
# 保证两方通信
server.send(torch.tensor(1))


def print_original_data():
    print("===============================================")
    print("X:", X)
    print("Y:", Y)
    print("T:", T)

    print("===============================================")


def gelu_test(x):
    # print("===============================================")
    # gelu = torch.nn.GELU()
    # print("torch.GELU:", gelu(X))

    secGelu = SecGeLU()
    share_z = secGelu.forward(x)
    # print("返回结果", share_z.restore().convert_to_real_field())

    # print("===============================================")


def div_test(x, y):
    # print("===============================================")
    # print(X / Y)
    share_z = x / y
    # print(share_z.restore().convert_to_real_field())
    # print("================================================")


def exp_test(x):
    # print("===============================================")
    # print("明文数据", torch.exp(X))
    share_z = ArithmeticSecretSharing.exp(x)
    # print(share_z.restore().convert_to_real_field())
    # print("===============================================")
    #
    # print("明文数据先exp再求和", torch.sum(torch.exp(X), dim=-1))
    # share_z = ArithmeticSecretSharing.exp(x)
    # share_z = share_z.sum(dim=-1)
    # print("密文求和", share_z.restore().convert_to_real_field())
    # print("===============================================")


def inv_sqrt_test(x):
    # print("===============================================")
    # print("明文数据", torch.rsqrt(Y))
    share_z = ArithmeticSecretSharing.rsqrt(x)
    # print(share_z.restore().convert_to_real_field())
    # print(torch.max((torch.rsqrt(Y)) - share_z.restore().convert_to_real_field()))
    # print("===============================================")


def softmax_test(x):
    # print("===============================================")
    # softmax = torch.nn.Softmax(dim=-1)
    # print("torch.softmax:", softmax(Y))

    secSoftmax = SecSoftmax(dim=-1)
    share_z = secSoftmax.forward(x)
    # print("返回结果", share_z.restore().convert_to_real_field())

    # print("===============================================")


def layer_norm_test(x):
    # print("===============================================")
    # layernorm = torch.nn.LayerNorm(4)
    # layernorm.cuda()
    # print("torch.LayerNorm:", layernorm(Y))

    secLayerNorm = SecLayerNorm(100)
    share_z = secLayerNorm.forward(x)

    # print("返回结果", share_z.restore().convert_to_real_field())

    # print("===============================================")


def tanh_test(x):
    # print("===============================================")
    # tanh = torch.nn.Tanh()
    # print("torch.Tanh:", tanh(X))

    secTanh = SecTanh()
    share_z = secTanh.forward(x)
    # print("返回结果", share_z.restore().convert_to_real_field())

    # print("===============================================")


if __name__ == '__main__':
    print_original_data()
    comm_count(server.communicator, get_time, gelu_test, share_x)
    torch.cuda.empty_cache()

    comm_count(server.communicator, get_time, exp_test, share_y)
    torch.cuda.empty_cache()

    comm_count(server.communicator, get_time, div_test, share_x, share_y)
    torch.cuda.empty_cache()

    comm_count(server.communicator, get_time, softmax_test, share_y)
    torch.cuda.empty_cache()

    comm_count(server.communicator, get_time, inv_sqrt_test, share_y)
    torch.cuda.empty_cache()

    comm_count(server.communicator, get_time, layer_norm_test, share_x)
    torch.cuda.empty_cache()

    comm_count(server.communicator, get_time, tanh_test, share_x)
    torch.cuda.empty_cache()
