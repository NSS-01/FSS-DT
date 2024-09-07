"""
使用同态加密生成beaver三元组和msb三元组
"""

from crypto.mpc.semi_honest_party import SemiHonestCS
from crypto.protocols.parameter_buffer.buffers import MSBBuffer
from crypto.primitives.beaver.beaver_triples import BeaverBuffer

client = SemiHonestCS(type='client')
client.set_address('127.0.0.1')
client.set_port(20000)
client.connect()

beaver_provider = BeaverBuffer(client)
msb_provider = MSBBuffer(client)

beaver_provider.gen_beaver_triples_by_homomorphic_encryption(num_of_triples=10)
msb_provider.gen_msb_triples_by_homomorphic_encryption(num_of_triples=10)
