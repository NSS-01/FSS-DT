from crypto.mpc.semi_honest_party import SemiHonestCS
from crypto.protocols.parameter_buffer.buffers import MSBBuffer
from crypto.primitives.beaver.beaver_triples import BeaverBuffer

server = SemiHonestCS(type='server')
server.set_address('127.0.0.1')
server.set_port(20000)
server.connect()

beaver_provider = BeaverBuffer(server)
msb_provider = MSBBuffer(server)

beaver_provider.gen_beaver_triples_by_homomorphic_encryption(num_of_triples=10)
msb_provider.gen_msb_triples_by_homomorphic_encryption(num_of_triples=10)
