from application.decision_tree.mostree import compare_threshold
from crypto.protocols.RSS_malicious_subprotocol.protocols import *
from crypto.mpc.malicious_party import Malicious3PCParty
from crypto.primitives.oblivious_transfer.oblivious_select_dpf import ObliviousSelect
from crypto.primitives.oblivious_transfer.msb_with_os import MSBWithOS

from crypto.protocols.CMP.verifiable_sigma import *

party_id = 0

Party = Malicious3PCParty(party_id=party_id)
Party.online()
Party.generate_prg_seed()
Party.beaver_provider.load_param(3)
Party.key_provider.load_keys(3)
Party.key_provider.load_self_keys()
Party.beaver_provider_2pc.load_param()

os = ObliviousSelect(Party)
cmp_os = MSBWithOS(Party)
num = 10

x = torch.randint(-5, 5, [num], dtype=data_type, device=DEVICE)
print(x)
x = RingTensor.convert_to_ring(x)

X = share(x, Party)

cmp_os.preprocess(10)
print("preprocess done")
start = time.time()
res = cmp_os.msb(X)
end = time.time()
print(end - start)
open_res = open(res)
print(open_res)
# print((open_res == x.signbit() + 0).sum() - num)

start = time.time()
compare_threshold(0, X, X, 0)
end = time.time()
print(end - start)

# table = torch.tensor([[0, 1, 2, 3, 4], [0, 2, 3, 4, 1]], dtype=data_type, device=DEVICE)
# table = RingTensor.convert_to_ring(table)
# table = share(table, Party)
#
# os.preprocess(10)
# print("preprocess done")
# res = os.selection(table, X)
#
# print(open(res))
