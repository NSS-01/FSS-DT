# test client

from crypto.mpc.semi_honest_party import SemiHonestCS
from crypto.primitives.arithmetic_secret_sharing.arithmetic_secret_sharing import ArithmeticSecretSharing
from debug.crypto.mpc.party.configs import ADDRESS, PORT, DTYPE, SCALE

client = SemiHonestCS(type='client')
client.set_dtype(DTYPE)
client.set_scale(SCALE)
client.connect(('127.0.0.1', 20000), ('127.0.0.1', 20001), ('127.0.0.1', 8089), ('127.0.0.1', 8088))

# receive other shares from server
shared_x = ArithmeticSecretSharing(client.receive_ring_tensor(), client)
print(shared_x)
print(shared_x.restore())

# receive other shares from server
# shared_y = ArithmeticSecretSharing(client.receive_ring_tensor(), client)
# print(shared_y)
# print(shared_y.restore())
#
# z_shared = shared_x * shared_y
# print(z_shared)
# print(z_shared.restore())
