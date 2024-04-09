from application.decision_tree.mostree import compare_threshold
from crypto.primitives.oblivious_transfer.msb_with_os import MSBWithOS
from crypto.primitives.oblivious_transfer.oblivious_select_dpf import ObliviousSelect
from crypto.protocols.RSS_malicious_subprotocol.protocols import *
from crypto.mpc.malicious_party import Malicious3PCParty

party_id = 1

Party = Malicious3PCParty(party_id=party_id)
Party.online()
Party.generate_prg_seed()
Party.beaver_provider.load_param(3)
Party.key_provider.load_keys(3)
Party.key_provider.load_self_keys()
Party.beaver_provider_2pc.load_param()

os = ObliviousSelect(Party)
cmp_os = MSBWithOS(Party)

X = receive_share_from(input_id=0, party=Party)

cmp_os.preprocess(10)
print("preprocess done")
start = time.time()
res = cmp_os.msb(X)
end = time.time()
print(open(res))

start = time.time()
compare_threshold(0, X, X, 0)
end = time.time()
print(end - start)

# table = receive_share_from(input_id=0, party=Party)
# os.preprocess(10)
#
# res = os.selection(table, X)
#
# print(open(res))
