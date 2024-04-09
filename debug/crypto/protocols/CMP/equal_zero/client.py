from config.base_configs import *
from config.network_config import TEST_SERVER_ADDRESS, TEST_SERVER_PORT
from crypto.mpc.semi_honest_party import SemiHonestCS
from crypto.primitives.arithmetic_secret_sharing.arithmetic_secret_sharing import ArithmeticSecretSharing
from crypto.primitives.function_secret_sharing.dpf import DPFKey
from crypto.protocols.CMP.multi_rounds_equal_match import equal_match_key_gen_all_dpf, equal_match_all_dpf, \
    equal_match_one_msb
from crypto.protocols.providers.beaver_provider import BeaverProvider
from crypto.tensor.RingTensor import RingTensor

from crypto.protocols.encoding.encoding import *

x = RingTensor.convert_to_ring(6)
# x.bit_len = 4
x_encoding, x_mask = zero_encoding(x)



# y = RingTensor.convert_to_ring(10)
# y_encoding, _ = one_encoding(y)


client = SemiHonestCS(type='client')
client.set_address(TEST_SERVER_ADDRESS)
client.set_port(TEST_SERVER_PORT)
client.set_dtype(DTYPE)
client.set_scale(SCALE)
client.set_beaver_provider(BeaverProvider(client))
# server.set_compare_key_provider()
client.beaver_provider.load_triples(2)
client.connect()


x_encoding_shared = ArithmeticSecretSharing(x_encoding, client)
y_encoding_shared = ArithmeticSecretSharing(RingTensor.convert_to_ring(0), client)


# print(x_encoding_shared.restore())
# print(y_encoding_shared.restore())

d = x_encoding_shared - y_encoding_shared


# print(d.restore())

# exit()
# number_of_keys = 10
# test_input = torch.randint(-3, 10, [number_of_keys])
# # test_input = torch.zeros([number_of_keys], dtype=torch.int64)
# y = RingTensor.convert_to_ring(test_input)

# K0, K1 = equal_key_gen(16, number_of_keys)

# y0, y1 = ArithmeticSecretSharing.share(y, 2)

# k11, r11, k12, r12 = K1

k1 = DPFKey.dic_to_key(client.receive_params())
k2 = DPFKey.dic_to_key(client.receive_params())
k3 = DPFKey.dic_to_key(client.receive_params())
r1 = client.receive_ring_tensor()
r2 = client.receive_ring_tensor()
r3 = client.receive_ring_tensor()
client.send_tensor(torch.tensor(0))

# y1 = client.receive_ring_tensor()

# x = ArithmeticSecretSharing(d[0], client)

res = equal_match_all_dpf(x_encoding_shared, y_encoding_shared, (k1, r1, k2, r2, k3, r3), 8)

print(res.restore())

res = equal_match_one_msb(x_encoding_shared, y_encoding_shared, k1, r1, (k2, r2, k3, r3))

print(res.restore())
