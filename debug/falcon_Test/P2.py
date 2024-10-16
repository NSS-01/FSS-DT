from NssMPC.secure_model.mpc_party import HonestMajorityParty
# from functional import *
from functional_our import *
id = 2
Party = HonestMajorityParty(id=id)
Party.set_comparison_provider()
Party.set_multiplication_provider()
Party.set_trunc_provider()
Party.online()

# testMatMul(784, 128, 10, 1, Party)
testMatMul(1000, 256, 100, 10, Party)