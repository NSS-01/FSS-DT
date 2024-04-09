from crypto.mpc.semi_honest_party import SemiHonestCS
from crypto.protocols.exp.exp import exp_eval
from crypto.tensor.ArithmeticSharedRingTensor import ArithmeticSharedRingTensor
from crypto.primitives.beaver.beaver_triples import BeaverBuffer

client = SemiHonestCS(type='client')
client.set_beaver_provider(BeaverBuffer(client))
client.set_wrap_provider()
client.set_compare_key_provider()
client.set_neg_exp_provider()
client.beaver_provider.load_param()
client.connect(('127.0.0.1', 20000), ('127.0.0.1', 20001), ('127.0.0.1', 8089), ('127.0.0.1', 8088))

k1 = client.receive()

x1 = client.receive()
x1 = ArithmeticSharedRingTensor(x1, client)

res = exp_eval(x1, k1)

print(res.restore().convert_to_real_field())
