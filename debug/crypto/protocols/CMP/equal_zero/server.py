import time

from config.base_configs import *
from config.network_config import TEST_SERVER_ADDRESS, TEST_SERVER_PORT
from crypto.mpc.semi_honest_party import SemiHonestCS
from crypto.primitives.arithmetic_secret_sharing.arithmetic_secret_sharing import ArithmeticSecretSharing
from crypto.protocols.CMP.multi_rounds_equal_match import *
from crypto.protocols.providers.beaver_provider import BeaverProvider
from crypto.tensor.RingTensor import RingTensor

from crypto.protocols.encoding.encoding import *

# x = RingTensor.convert_to_ring(6)
# x_encoding, _ = zero_encoding(x)

y = RingTensor.convert_to_ring(10)
# y.bit_len = 4
y_encoding, y_mask = one_encoding(y)

server = SemiHonestCS(type='server')
server.set_address(TEST_SERVER_ADDRESS)
server.set_port(TEST_SERVER_PORT)
server.set_dtype(DTYPE)
server.set_scale(SCALE)
server.set_beaver_provider(BeaverProvider(server))
# server.set_compare_key_provider()
server.beaver_provider.load_triples(2)
server.connect()

x_encoding_shared = ArithmeticSecretSharing(RingTensor.convert_to_ring(0), server)
y_encoding_shared = ArithmeticSecretSharing(y_encoding, server)

# print(x_encoding_shared.restore())
# print(y_encoding_shared.restore())

d = x_encoding_shared - y_encoding_shared

# print(d.restore())

# exit()
number_of_keys = BIT_LEN
# test_input = torch.randint(-3, 10, [number_of_keys])
# # test_input = torch.zeros([number_of_keys], dtype=torch.int64)
# y = RingTensor.convert_to_ring(test_input)
# print(y)

K0, K1 = equal_match_key_gen_all_dpf(8, number_of_keys)

# y0, y1 = ArithmeticSecretSharing.share(y, 2)

k01, r01, k02, r02, k03, r03 = K0
k11, r11, k12, r12, k13, r13 = K1

server.send_params(k11)
server.send_params(k12)
server.send_params(k13)
server.send_ring_tensor(r11)
server.send_ring_tensor(r12)
server.send_ring_tensor(r13)
server.receive_tensor()

# server.send_ring_tensor(y1)

# x = ArithmeticSecretSharing(d[0], server)

start = time.time()
res = equal_match_all_dpf(x_encoding_shared, y_encoding_shared, K0, 8)
end = time.time()
print(end - start)

print(res.restore())

start = time.time()
res = equal_match_one_msb(x_encoding_shared, y_encoding_shared, k01, r01, (k02, r02, k03, r03))
end = time.time()
print(end - start)

print(res.restore())
