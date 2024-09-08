import torch

BIT_LEN = 64
RING_MAX = 2 ** BIT_LEN
HALF_RING = 2 ** (BIT_LEN - 1)
LAMBDA = 128
# LAMBDA = 127

GE_TYPE = 'SIGMA'
PRG_TYPE = 'AES'
#DEVICE = 'cuda'
DEVICE = 'cpu'
DEBUG = True
# DEBUG = False

assert BIT_LEN in (64, 32)
assert GE_TYPE in ('MSB', 'FSS', 'GROTTO', 'SIGMA')
assert DEVICE in ('cuda', 'cpu', 'cuda:0', 'cuda:1')
data_type = torch.int64 if BIT_LEN == 64 else torch.int32  # 目前仅支持BIT_LEN=64和32，TODO：放在这是否合适

# 定点数设置
DTYPE = 'int'  # 'float' or 'int
float_scale = 65536 if BIT_LEN == 64 else 127  # 64位的环上浮点数默认精度为65536，32位环上默认精度为127
int_scale = 1
SCALE = float_scale if DTYPE == 'float' else int_scale

DTYPE_MAPPING = {
    torch.int32: int_scale,
    torch.int64: int_scale,
    torch.float32: float_scale,
    torch.float64: float_scale,
    'int': int_scale,
    'float': float_scale,
}
DTYPE_SCALE_MAPPING = {
    'int': int_scale,
    'float': float_scale,
}
base_path = './data/{}'.format(BIT_LEN)
model_file_path = base_path + '/NN/'
triple_path = base_path + '/triples_data/'
sigma_key_path = base_path + '/cmp_sigma_keys/'
msb_data_path = base_path + '/MSB/triples_data/'
edc_auxiliary_parameters_path = base_path + '/edc/auxiliary_parameters/'

SOCKET_MAX_SIZE = 32 * 1024

SOCKET_NUM = 1
