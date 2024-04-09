from application.decision_tree.decision_tree import DecisionTree
from application.decision_tree.mostree import MosTree
from common.utils.timer import get_time
from crypto.protocols.RSS_malicious_subprotocol.protocols import *
from crypto.mpc.malicious_party_new import Malicious3PCParty

from debug.application.DT.preparation import train_datasets

clf = train_datasets('wine')
# clf = train_datasets('breast_cancer')
# clf = train_datasets('digits')
# clf = train_datasets('spambase')
# clf = train_datasets('diabetes')
# clf = train_datasets('boston')
# clf = train_datasets('mnist')

party_id = 0

Party = Malicious3PCParty(party_id=party_id)
Party.online()
Party.generate_prg_seed()
Party.os_cmp.preprocess(10)
Party.os_dpf.preprocess(10)
Party.beaver_provider.load_param(3)
Party.beaver_provider_2pc.load_param()

x = receive_share_from(1, Party)
print("===================our work=====================")
dt = DecisionTree(Party)
dt.setup(clf)
get_time(dt.evaluation, x)

print("===================mostree=====================")
dt = MosTree(Party)
dt.setup(clf)
get_time(dt.evaluation, x)

Party.close()
