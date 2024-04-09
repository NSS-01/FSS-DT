from application.decision_tree.decision_tree import DecisionTree
from application.decision_tree.mostree import MosTree
from common.utils.timer import get_time
from crypto.protocols.RSS_malicious_subprotocol.protocols import *
from crypto.mpc.malicious_party_new import Malicious3PCParty
from debug.application.DT.preparation import load_datasets

party_id = 1
Party = Malicious3PCParty(party_id=party_id)

Party.online()
Party.generate_prg_seed()
Party.os_cmp.preprocess(10)
Party.os_dpf.preprocess(10)
Party.beaver_provider.load_param(3)
Party.beaver_provider_2pc.load_param()

data, _ = load_datasets('wine')
# data, _ = load_datasets('breast_cancer')
# data, _ = load_datasets('digits')
# data, _ = load_datasets('spambase')
# data, _ = load_datasets('diabetes')
# data, _ = load_datasets('boston')

x = data[[0], :]
# x = data.data[[0, 1, 50, 51, 100, 177], :]
# x = data.data
x = torch.tensor(x.tolist(), device=DEVICE).to(data_type)
print(x)

# x = torch.ones([1, 784], dtype=data_type, device=DEVICE)  # mnist
x = RingTensor.convert_to_ring(x)
x = share(x, Party)

print("===================our work=====================")
dt = DecisionTree(Party)
dt.setup()

res = get_time(dt.evaluation, x)
print(res)

print("===================mostree=====================")
dt = MosTree(Party)
dt.setup()
start = time.time()
res = get_time(dt.evaluation, x)
print(res)

Party.close()
