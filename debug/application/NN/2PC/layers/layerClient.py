'''
安全两方计算下的神经网络每层的测试
'''
import torch
from crypto.primitives.arithmetic_secret_sharing.arithmetic_secret_sharing import ArithmeticSecretSharing
from crypto.primitives.beaver.beaver import BeaverOfflineProvider
from crypto.mpc.semi_honest_party import SemiHonestCS
from crypto.tensor.RingTensor import RingTensor
from config.base_configs import *
from config.network_config import *
import application.NN.layers.layer as layer

client = SemiHonestCS(type='client')
client.set_address(TEST_CLIENT_ADDRESS)
client.set_port(TEST_CLIENT_PORT)
# client.set_dtype('int')
# client.set_scale(1)
client.set_dtype(DTYPE)
client.set_scale(SCALE)
client.set_beaver_provider(BeaverOfflineProvider())
client.set_compare_key_provider()
client.beaver_provider.load_param(client, 2)
client.connect()

x_1 = client.receive_ring_tensor()
share_x = ArithmeticSecretSharing(x_1, client)

epsilon11 = client.receive_ring_tensor()
epsilon21 = client.receive_ring_tensor()

# w_1 = client.receive_ring_tensor()
#
# client.receive_tensor()
# print("=====================conv===============================")
# conv = layer.SecConv2d(weight=w_1)
# res_x = conv(share_x)
# print(res_x.restore().convert_to_real_field())
# print("======================maxPooling==============================")
# maxPool = layer.SecMaxPool2D(kernel_shape=2, stride=2)
# res_x = maxPool(share_x)
# print(res_x.restore().convert_to_real_field())
# print("=======================relu=============================")
# relu = layer.SecReLu()
# res_x = relu(share_x)
# print(res_x.restore().convert_to_real_field())
# # client.receive_tensor()
# # print("==========================gemm==========================")
# # gemm = layer.SecGemm(weight=w_linear_1, bias=b_linear_1)
# # share_x = gemm(share_x)
# # print(share_x.restore().convert_to_real_field())
# client.receive_tensor()
# print("=====================avgPooling===============================")
# avgPool = layer.SecAvgPool2D(kernel_shape=2, stride=2)
# res_x = avgPool(share_x)
# print(res_x.restore().convert_to_real_field())
# client.receive_tensor()
# print("=====================adaptiveAvgPooling===============================")
# avgPool = layer.SecAdaptiveAvgPool2D(output_size=(1, 1))
# res_x = avgPool(share_x)
# print(res_x.restore().convert_to_real_field())
# client.receive_tensor()
print("=====================batchNormalization===============================")
batchNorm = layer.SecBatchNorm2D(epsilon11, epsilon21)
res_x = batchNorm(share_x)
print(res_x.restore().convert_to_real_field())
print("====================================================")