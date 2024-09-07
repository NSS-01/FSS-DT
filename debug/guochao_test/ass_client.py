"""
ASS基础功能测试
"""
from application.NN.NNCS import NeuralNetworkCS
from application.NN.layers.SecLayer import *
from config.base_configs import *
from crypto.mpc.semi_honest_party import SemiHonestCS
from crypto.tensor.ArithmeticSharedRingTensor import ArithmeticSharedRingTensor
from crypto.primitives.beaver.beaver_triples import BeaverBuffer
from crypto.tensor.RingTensor import RingTensor

client = NeuralNetworkCS(type='client')
client.inference_transformer(is_transformer=True)
client.connect(('127.0.0.1', 20000), ('127.0.0.1', 20001), ('127.0.0.1', 8089), ('127.0.0.1', 8088))

x_1 = client.receive()
shared_x = ArithmeticSharedRingTensor(x_1, client)

y_1 = client.receive()
shared_y = ArithmeticSharedRingTensor(y_1, client)

t_1 = client.receive()
shared_t = ArithmeticSharedRingTensor(t_1, client)
client.receive()


# 打开代码测试相应功能

def test_sec_gelu(x):
    s_gelu = SecGELU()
    share_z = s_gelu.forward(x)
    share_z.restore().convert_to_real_field()


def test_div(x, y):
    print("===================================")
    share_z = x / y
    print(share_z.restore().convert_to_real_field())
    print("====================================")


# def test_pow(t):
#     share_res = t ** 7
#     print(share_res.restore().convert_to_real_field())


# def test_div_truth(x):
#     share_z = secure_div_constant(x, 1.4142)
#     print(share_z.restore().convert_to_real_field())


def test_exp(x):
    share_z = ArithmeticSharedRingTensor.nexp(x)
    share_z.restore().convert_to_real_field()

    share_z = ArithmeticSharedRingTensor.nexp(x)
    share_z = share_z.sum(dim=-1)
    print(share_z.restore().convert_to_real_field())


def test_inv_sqrt(x):
    share_z = ArithmeticSharedRingTensor.reciprocal_sqrt(x)
    share_z.restore().convert_to_real_field()
    share_z.restore().convert_to_real_field()


def test_sec_softmax(x):
    sec_softmax = SecSoftmax()
    share_z = sec_softmax.forward(x)
    share_z.restore().convert_to_real_field()


def test_layer_norm(x):
    sec_layer_norm = SecLayerNorm(4)
    share_z = sec_layer_norm.forward(x)
    share_z.restore().convert_to_real_field()


def test_embedding():
    print("===============================================")
    x1 = client.receive()
    share_x = ArithmeticSharedRingTensor(x1, client)

    embedding = torch.nn.Embedding(10, 4)

    weight = client.receive()

    secembedding = SecEmbedding(10, 4, _weight=weight)
    share_z = secembedding.forward(share_x)

    print("返回结果", share_z.restore().convert_to_real_field())

    print("===============================================")


if __name__ == '__main__':
    # test_div(shared_x, shared_y)
    # # test_div_truth(shared_x)
    # test_sec_gelu(shared_x)
    # # test_pow(shared_t)
    # test_exp(shared_x)
    # test_inv_sqrt(shared_x)
    # test_sec_softmax(shared_x)
    test_layer_norm(shared_x)
    test_embedding()
