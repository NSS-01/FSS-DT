from application.decision_tree.decision_tree import DecisionTree
from application.decision_tree.mostree import MosTree
from common.utils.timer import get_time
from crypto.protocols.RSS_malicious_subprotocol.protocols import *
from crypto.mpc.malicious_party_new import Malicious3PCParty

party_id = 2
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
dt.setup()
get_time(dt.evaluation, x)

print("===================mostree=====================")
dt = MosTree(Party)
dt.setup()
get_time(dt.evaluation, x)

Party.close()
