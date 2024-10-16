import torch

from NssMPC import RingTensor
from NssMPC.config import data_type, DEVICE
from NssMPC.crypto.primitives.oblivious_transfer.oblivious_select_dpf import ObliviousSelect
from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional import open, receive_share_from
from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional.msb_with_os import MSBWithOS
from NssMPC.secure_model.mpc_party import HonestMajorityParty

Party = HonestMajorityParty(2)
# Party.set_oblivious_selection_provider()
Party.set_conditional_oblivious_selection_provider()
Party.online()
# table = receive_share_from(0, Party)
X = receive_share_from(0, Party)
print("preprocess done")
# res = ObliviousSelect.selection(table, X)
# print(open(res))
# res = ObliviousSelect.selection(table, X)
# print(open(res))
# res = ObliviousSelect.selection(table, X)
# print(open(res))
res = MSBWithOS.msb_without_mac_check(X)
print(open(res[0]))
