from NssMPC.config.configs import SCALE
from NssMPC.common.ring import *
from NssMPC.crypto.aux_parameter.truncation_keys.ass_trunc_aux_param import AssTruncAuxParams


def truncate(share):
    # 使用ABY3的trunc2进行计算
    # 该方法的核心思想是使用一个预先计算的r和其trunc值r'对秘密分享的数据进行盲化，然后恢复成明文 z = x + r，随后再进行计算
    # z' = trunc(z) 最后再用秘密分享的r'对其进行恢复，得到 x' = z' - r'
    # 这一计算步骤使得截断操作发生在明文上，因此不用考虑秘密分享恢复的超环问题，但是仍然存在 x+r的超环问题
    # 因此本实现限定了r的取值范围和x的值域范围 使得 x+r < RingMax
    print("trunc")
    r, r_t = share.party.get_param(AssTruncAuxParams, share.numel())
    shape = share.shape
    share = share.flatten()
    r.party = share.party
    r_t.party = share.party
    r_t.dtype = 'float'
    r.dtype = 'float'
    delta_share = share - r
    delta = delta_share.restore()
    delta_trunc = delta // SCALE
    result = r_t + delta_trunc
    return result.reshape(shape)


