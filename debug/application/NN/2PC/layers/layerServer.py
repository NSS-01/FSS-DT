'''
安全两方计算下的神经网络每层的测试
'''

from crypto.primitives.arithmetic_secret_sharing.arithmetic_secret_sharing import ArithmeticSecretSharing
from crypto.mpc.semi_honest_party import SemiHonestCS
from crypto.primitives.beaver.beaver_triples import BeaverFactory
from crypto.tensor.RingTensor import RingTensor
from config.base_configs import *
from config.network_config import *
import application.NN.layers.layer as layer

server = SemiHonestCS(type='server')
server.set_address(TEST_SERVER_ADDRESS)
server.set_port(TEST_SERVER_PORT)
# server.set_dtype('int')
# server.set_scale(1)
server.set_dtype(DTYPE)
server.set_scale(SCALE)
server.set_beaver_provider(BeaverFactory())
server.set_compare_key_provider()
server.beaver_provider.load_param(server, 2)
server.connect()

x = torch.rand([1, 10, 2, 2])
print(x)
x_ring = RingTensor.convert_to_ring(x)

x_0, x_1 = ArithmeticSecretSharing.share(x_ring, 2)
server.send_ring_tensor(x_1)
share_x = ArithmeticSecretSharing(x_0, server)
share_x.ring_tensor.dtype = server.dtype
share_x.ring_tensor.scale = server.scale


m = torch.nn.BatchNorm2d(num_features=10)
m = m.eval()
res = m(x)
print(res)

gamma = m.weight
beta = m.bias
running_mean = m.running_mean
running_variance = m.running_var

epsilon1 = gamma / running_variance
epsilon2 = beta - (gamma * running_mean) / running_variance

epsilon10, epsilon11 = ArithmeticSecretSharing.share(RingTensor.convert_to_ring(epsilon1), 2)
epsilon20, epsilon21 = ArithmeticSecretSharing.share(RingTensor.convert_to_ring(epsilon2), 2)

server.send_ring_tensor(epsilon11)
server.send_ring_tensor(epsilon21)
#
# epsilon1 = epsilon1.unsqueeze(1)
# epsilon1 = epsilon1.unsqueeze(2)
#
# epsilon2 = epsilon2.unsqueeze(1)
# epsilon2 = epsilon2.unsqueeze(2)
#
# t = epsilon1 * x + epsilon2
#
# print(t)
# print(t.shape)

# m = torch.nn.AdaptiveAvgPool2d((1, 1))
# m = m.eval()
# res = m(x)
# print(res)

# kernel_size = 3
# w = torch.rand([1, 1, kernel_size, kernel_size])
# w_0, w_1 = ArithmeticSecretSharing.share(RingTensor.convert_to_ring(w), 2)
# server.send_ring_tensor(w_1)
#
# beaver_for_conv(x, w, 0, 1, server)
# server.send_tensor(torch.tensor(1))
# print("=====================conv===============================")
# start = time.time()
# conv = layer.SecConv2d(weight=w_0)
# res_x = conv(share_x)
# end = time.time()
# print(end - start)
# print(res_x.restore().convert_to_real_field())
# print("=====================maxPooling===============================")
# start = time.time()
# maxPool = layer.SecMaxPool2D(kernel_shape=2, stride=2)
# res_x = maxPool(share_x)
# end = time.time()
# print(end - start)
# print(res_x.restore().convert_to_real_field())
# print("=======================relu=============================")
# start = time.time()
# relu = layer.SecReLu()
# res_x = relu(share_x)
# end = time.time()
# print(end - start)
# print(res_x.restore().convert_to_real_field())
# beaver_for_conv(x, kernel_size, 0, 1, server)
# server.send_tensor(torch.tensor(1))
# print("========================gemm============================")
# start = time.time()
# gemm = layer.SecGemm(weight=w_linear_0, bias=b_linear_0)
# share_x = gemm(share_x)
# end = time.time()
# print(end - start)
# print(share_x.restore().convert_to_real_field())
# beaver_for_avg_pooling(x, 2, 0, 2, server)
# server.send_tensor(torch.tensor(1))
# print("=====================avgPooling===============================")
# start = time.time()
# avgPool = layer.SecAvgPool2D(kernel_shape=2, stride=2)
# res_x = avgPool(share_x)
# end = time.time()
# print(end - start)
# print(res_x.restore().convert_to_real_field())
# beaver_for_adaptive_avg_pooling(x, (1, 1), server)
# server.send_tensor(torch.tensor(1))
# print("=====================adaptiveAvgPooling===============================")
# start = time.time()
# avgPool = layer.SecAdaptiveAvgPool2D(output_size=(1, 1))
# end = time.time()
# print(end - start)
# res_x = avgPool(share_x)
# print(res_x.restore().convert_to_real_field())
# beaver_for_batch_normalization(x, epsilon1, server)
# server.send_tensor(torch.tensor(1))
print("=====================batchNormalization===============================")
batchNorm = layer.SecBatchNorm2D(epsilon10, epsilon20)
res_x = batchNorm(share_x)
ans = res_x.restore().convert_to_real_field()
print(ans)
print(ans.shape)
print("====================================================")
