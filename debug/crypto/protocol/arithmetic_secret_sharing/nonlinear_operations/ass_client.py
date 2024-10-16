import torch

from NssMPC import ArithmeticSecretSharing
from application.neural_network.layers import SecGeLU, SecSoftmax, SecLayerNorm, SecTanh
from application.neural_network.party import NeuralNetworkCS

client = NeuralNetworkCS(type='secure_version')
client.activate_by_gelu()
client.online()

x_1 = client.receive()
shared_x = x_1

y_1 = client.receive()
shared_y = y_1

t_1 = client.receive()
shared_t = t_1
client.receive()


# 打开代码测试相应功能

def sec_gelu_test(x):
    s_gelu = SecGeLU()
    share_z = s_gelu.forward(x)
    # share_z.restore().convert_to_real_field()


def div_test(x, y):
    # print("===================================")
    share_z = x / y
    # print(share_z.restore().convert_to_real_field())
    # print("====================================")


def exp_test(x):
    share_z = ArithmeticSecretSharing.exp(x)
    # share_z.restore().convert_to_real_field()
    #
    # share_z = ArithmeticSecretSharing.exp(x)
    # share_z = share_z.sum(dim=-1)
    # print(share_z.restore().convert_to_real_field())


def inv_sqrt_test(x):
    share_z = ArithmeticSecretSharing.rsqrt(x)
    # share_z.restore().convert_to_real_field()
    # share_z.restore().convert_to_real_field()


def sec_softmax_test(x):
    sec_softmax = SecSoftmax()
    share_z = sec_softmax.forward(x)
    # share_z.restore().convert_to_real_field()


def layer_norm_test(x):
    sec_layer_norm = SecLayerNorm(100)
    share_z = sec_layer_norm.forward(x)
    # share_z.restore().convert_to_real_field()


def tanh_test(x):
    sec_tanh = SecTanh()
    share_z = sec_tanh.forward(x)
    # print("返回结果", share_z.restore().convert_to_real_field())


if __name__ == '__main__':
    sec_gelu_test(shared_x)
    torch.cuda.empty_cache()

    exp_test(shared_y)
    torch.cuda.empty_cache()

    div_test(shared_x, shared_y)
    torch.cuda.empty_cache()

    sec_softmax_test(shared_y)
    torch.cuda.empty_cache()

    inv_sqrt_test(shared_y)
    torch.cuda.empty_cache()

    layer_norm_test(shared_x)
    torch.cuda.empty_cache()

    tanh_test(shared_x)
    torch.cuda.empty_cache()
