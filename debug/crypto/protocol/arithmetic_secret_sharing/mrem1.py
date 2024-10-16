from crypto.protocols.beaver.arithmetic_triples import MulTriples
from NssMPC.crypto.protocols import equal_match_all_dpf
from model.mpc_party.semi_honest import SemiHonestCS

client = SemiHonestCS(type='secure_version')
client.set_multiplication_provider()
client.set_comparison_provider()
client.providers[MulTriples].load_param()
client.connect(('127.0.0.1', 20010), ('127.0.0.1', 20011), ('127.0.0.1', 8189), ('127.0.0.1', 8188))

x1 = client.receive()
y1 = client.receive()
k1 = client.receive()

# z1 = equal_match_one_msb(ArithmeticSharedRingTensor(RingTensor(0), secure_version), y1 - x1, None, k1)
z1 = equal_match_all_dpf(x1, y1, k1, 8)

client.send(z1)
