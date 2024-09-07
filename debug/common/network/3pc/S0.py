from common.utils.timer import get_time
# from crypto.mpc.malicious_party import Malicious3PCParty
from crypto.mpc.malicious_party_new import Malicious3PCParty
import torch

party_id = 0
Party = Malicious3PCParty(party_id=party_id)
Party.online()

num = 1024 * 1024 * 32
a = torch.randint(0, 10, [num], dtype=torch.int64)

get_time(Party.send_torch_tensor_to, 1, a)
# Party.send_torch_tensor_to(1, a)
Party.send_torch_tensor_to(2, a)

Party.close()
