from application.NN.layers.complex_nonlinear_layers.sec_gelu import GeLUKey
from crypto.primitives.beaver.beaver_triples import BeaverTriples
from crypto.primitives.function_secret_sharing.dicf import DICFKey
from crypto.primitives.function_secret_sharing.p_dicf import ParityDICFKey
from crypto.primitives.msb.msb_triples import MSBTriples
from crypto.protocols.comparison.cmp_sigma import CMPSigmaKey
from crypto.protocols.exp.exp import ExpKey
from crypto.protocols.exp.pos_exp import PosExpKey
from crypto.protocols.exp.neg_exp import NegExpKey
from crypto.protocols.truncate.tr_crypten import Wrap

BeaverTriples.gen_and_save('TTP', 2, 10)
MSBTriples.gen_and_save(10, 2, 'TTP')
ParityDICFKey.gen_and_save(10)
DICFKey.gen_and_save(10)
Wrap.gen_and_save(10)
CMPSigmaKey.gen_and_save(10)
GeLUKey.gen_and_save(1)
NegExpKey.gen_and_save(1)
PosExpKey.gen_and_save(1)
ExpKey.gen_and_save(1)
