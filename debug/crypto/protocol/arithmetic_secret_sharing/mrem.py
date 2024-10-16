from NssMPC.common.ring.ring_tensor import RingTensor
from NssMPC.crypto.primitives.arithmetic_secret_sharing.arithmetic_secret_sharing import ArithmeticSecretSharing
from NssMPC.crypto.aux_parameter.beaver_triples import AssMulTriples
from NssMPC.crypto.protocols import zero_encoding, one_encoding
from NssMPC.crypto.protocols import equal_match_key_gen_all_dpf, \
    equal_match_all_dpf
from model.mpc_party.semi_honest import SemiHonestCS

server = SemiHonestCS(type='server')
server.set_multiplication_provider()
server.set_comparison_provider()
server.providers[AssMulTriples].load_param()

server.connect(('127.0.0.1', 8189), ('127.0.0.1', 8188), ('127.0.0.1', 20010), ('127.0.0.1', 20011))

x = RingTensor.convert_to_ring([1])

y = RingTensor.convert_to_ring([5])

bit_len = 8

x_encoding, x_mask = zero_encoding(x)
print(x_encoding)
# for encoding in x_encoding:
#     print(bin(encoding.tensor))
print("x_mask", x_mask)
#

y_encoding, y_mask = one_encoding(y)
print("y_encoding", y_encoding)
# for encoding in y_encoding:
#     print(bin(encoding.tensor))
print("y_mask", y_mask)

x = x_encoding
x0, x1 = ArithmeticSecretSharing.share(x, 2)
x0.party = server
server.send(x1)
y = y_encoding
y0, y1 = ArithmeticSecretSharing.share(y, 2)
y0.party = server
server.send(y1)

# k0, k1 = equal_match_key_gen_one_msb(len(x))
k0, k1 = equal_match_key_gen_all_dpf(bit_len, len(x))

server.send(k1)

# z0 = equal_match_one_msb(x0 - y0, ArithmeticSharedRingTensor(RingTensor(0), server), None, k0)
z0 = equal_match_all_dpf(x0, y0, k0, bit_len)

z1 = server.receive()

z = z0 + z1
print(z, "zzzzzzzzzzzzzzzz")
