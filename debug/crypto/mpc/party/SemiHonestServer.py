import torch

from crypto.mpc.semi_honest_party import SemiHonestCS
from crypto.primitives.arithmetic_secret_sharing.arithmetic_secret_sharing import ArithmeticSecretSharing
from crypto.tensor.RingTensor import RingTensor
from debug.crypto.mpc.party.configs import ADDRESS, PORT, DTYPE, SCALE

# from crypto.primitives.beaver.beaver import BeaverOfflineProvider

server = SemiHonestCS(type='server')
server.set_dtype(DTYPE)
server.set_scale(SCALE)
server.connect(('127.0.0.1', 8089), ('127.0.0.1', 8088), ('127.0.0.1', 20000), ('127.0.0.1', 20001))

# test arithmetic secret sharing
x = torch.tensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
x_ring = RingTensor.convert_to_ring(x)
x_0, x_1 = ArithmeticSecretSharing.share(x_ring, 2)
# send other shares to client
server.send_ring_tensor(x_1)
shared_x = ArithmeticSecretSharing(x_0, server)
print(shared_x)
print(shared_x.restore())

# test arithmetic secret sharing
# y = torch.tensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
# y_ring = RingTensor.convert_to_ring(y)
# y_0, y_1 = ArithmeticSecretSharing.share(y_ring, 2)
# # send other shares to client
# server.send_ring_tensor(y_1)
# shared_y = ArithmeticSecretSharing(y_0, server)
#
# print(shared_y)
# print(shared_y.restore())
#
# z = x * y
# print(z)
#
# z_shared = shared_x * shared_y
# print(z_shared)
# print(z_shared.restore())
