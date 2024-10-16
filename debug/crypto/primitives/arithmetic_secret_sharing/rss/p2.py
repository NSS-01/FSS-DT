from NssMPC import ReplicatedSecretSharing, RingTensor
from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional import v_matmul_with_trunc
from NssMPC.crypto.protocols.replicated_secret_sharing.semi_honest_functional import matmul_with_trunc
from NssMPC.secure_model.mpc_party import HonestMajorityParty, SemiHonest3PCParty
from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional.compare import secure_ge

id = 2

Party = HonestMajorityParty(id=id)
# Party = SemiHonest3PCParty(id=id)
Party.set_comparison_provider()
Party.set_multiplication_provider()
Party.set_trunc_provider()
Party.online()
share_table = ReplicatedSecretSharing([RingTensor([-1, -1]), RingTensor([2, 3])], Party)
print(share_table.restore())
shared_x = Party.receive(0)
shared_y = Party.receive(0)
print(shared_x.restore())


res = v_matmul_with_trunc(shared_x, shared_y)

print("res: ", res.restore().convert_to_real_field())


res_ori = shared_x @ shared_y
print("res_ori: ", res.restore().convert_to_real_field())
# print(shared_tensor)

# add_result = shared_tensor + shared_tensor
# res = add_result.restore()
#
# print(res)

# res = secure_ge(shared_x, shared_y)
# # mul_result = mul_result.view((-1, 1))
# res = res.restore()
# print(res.convert_to_real_field())
