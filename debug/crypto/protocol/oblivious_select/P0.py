import torch

from NssMPC import RingTensor
from NssMPC.config import data_type, DEVICE
from NssMPC.crypto.primitives.oblivious_transfer.oblivious_select_dpf import ObliviousSelect
from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional import share, open
from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional.msb_with_os import MSBWithOS
from NssMPC.secure_model.mpc_party import HonestMajorityParty

Party = HonestMajorityParty(0)
# Party.set_oblivious_selection_provider()
Party.set_conditional_oblivious_selection_provider()
Party.online()

# table = torch.tensor([[0, 1, 2, 3, 4], [0, 2, 3, 4, 1]], dtype=data_type, device=DEVICE)
# table = RingTensor.convert_to_ring(table)
# table = share(table, Party)
X = torch.tensor([-1, -9], dtype=data_type, device=DEVICE)
X = RingTensor.convert_to_ring(X)
X = share(X, Party)
print("preprocess done")
# res = ObliviousSelect.selection(table, X)
# print(open(res))
# res = ObliviousSelect.selection(table, X)
# print(open(res))
# res = ObliviousSelect.selection(table, X)
# print(open(res))
res = MSBWithOS.msb_without_mac_check(X)
print(open(res[0]))
