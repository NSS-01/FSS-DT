from NssMPC import RingTensor
from NssMPC.config import SCALE
from NssMPC.crypto.aux_parameter import RssMulTriples, RssMatmulTriples
from NssMPC.crypto.protocols.replicated_secret_sharing.semi_honest_functional import mul_with_out_trunc, matmul_with_out_trunc, reconstruct3out3, matmul_with_trunc
from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional.base import check_zero, open
from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional.truncate import truncate
from NssMPC.crypto.aux_parameter.truncation_keys.rss_trunc_aux_param import RssTruncAuxParams
def v_mul(x, y):
    """
    Multiplication of two RSS sharings ⟨x⟩ and ⟨y⟩
    :param x: an RSS sharing ⟨x⟩
    :param y: an RSS sharing ⟨y⟩
    :return: an RSS sharing ⟨x*y⟩
    """
    shape = x.shape if x.numel() >= y.numel() else y.shape
    x = x.expand(shape).flatten()
    y = y.expand(shape).flatten()
    ori_type = x.dtype
    res = mul_with_out_trunc(x, y)
    # print("单纯乘法：", res.restore())

    # a, b, c = x.party.providers[AssMulTriples].get_parameters(res.numel())
    a, b, c = x.party.get_param(RssMulTriples, res.numel())
    x_hat = x.clone()
    y_hat = y.clone()
    # print("c", c.restore())
    # c_tmp = mul_with_out_trunc(a, b)
    # print("c tmp", c_tmp.restore())

    # a = a.reshape(x.shape)
    # b = b.reshape(y.shape)
    # c = c.reshape(res.shape)
    # todo 临时方法，缺少原理论证
    a.dtype = b.dtype = c.dtype = x_hat.dtype = y_hat.dtype = 'int'
    e = x_hat + a
    f = y_hat + b
    e = open(e)
    f = open(f)
    # tmp1 = mul_with_out_trunc(b, e)
    check = -c + b * e + a * f - e * f
    # print("辅助参数乘法：", check.restore())
    check_zero(res + check)

    if ori_type == 'float':
        res = truncate(res)

    return res.reshape(shape)


def v_matmul(x, y):
    """
    Matrix Multiplication of two RSS sharings ⟨x⟩ and ⟨y⟩
    :param x: an RSS sharing ⟨x⟩
    :param y: an RSS sharing ⟨y⟩
    :aux_params: verify params
    :return: an RSS sharing ⟨x@y⟩

    """
    a, b, c = x.party.get_param(RssMatmulTriples, x.shape, y.shape)
    a.party = x.party
    b.party = x.party
    c.party = x.party

    ori_type = x.dtype
    res = matmul_with_out_trunc(x, y)

    x_hat = x.clone()
    y_hat = y.clone()
    # print("ori_res", res.restore())
    a.dtype = b.dtype = c.dtype = x_hat.dtype = y_hat.dtype = 'int'
    e = x_hat + a
    f = y_hat + b
    e = open(e)
    f = open(f)
    from NssMPC.crypto.primitives.arithmetic_secret_sharing.replicated_secret_sharing import ReplicatedSecretSharing
    mat_1 = ReplicatedSecretSharing([e @ b.item[0], e @ b.item[1]], x.party)
    mat_2 = ReplicatedSecretSharing([a.item[0] @ f, a.item[1] @ f], x.party)

    check = -c + mat_1 + mat_2 - e @ f
    check_zero(res + check)
    # print(ori_type)
    if ori_type == 'float':
        res = truncate(res)
    return res


def v_matmul_with_trunc(x, y):
    x_hat = x.clone()
    y_hat = y.clone()

    z_shared = \
        RingTensor.matmul(x.item[0], y.item[0]) + \
        RingTensor.matmul(x.item[0], y.item[1]) + \
        RingTensor.matmul(x.item[1], y.item[0])
    r, r_t = x.party.get_param(RssTruncAuxParams, z_shared.numel())
    shape = z_shared.shape
    share = z_shared.flatten()
    # todo：确定用这种方法的话需要改r r_t形式
    r_33 = r.item[0]
    # print("r_T", r_t)
    # r_33 = RingTensor(1)
    # print("r_33", r_33)
    r_t.party = x.party
    # todo:计算原理需要统一
    r_t.dtype = 'float'
    r.dtype = 'float'
    delta_share = z_shared - r_33
    delta = reconstruct3out3(delta_share, x.party)
    delta_trunc = delta // SCALE
    result = r_t + delta_trunc
    res = result.reshape(shape)


    a, b, c = x.party.get_param(RssMatmulTriples, x.shape, y.shape)
    a.party = x.party
    b.party = x.party
    c.party = x.party
    a.dtype = b.dtype = c.dtype = x_hat.dtype = y_hat.dtype = 'int'
    e = x_hat + a
    f = y_hat + b

    e_and_f = x.__class__.cat([e.flatten(), f.flatten()], dim=0)
    common_e_f = e_and_f.restore()
    e = common_e_f[:x.numel()].reshape(x.shape)
    f = common_e_f[x.numel():].reshape(y.shape)
    from NssMPC.crypto.primitives.arithmetic_secret_sharing.replicated_secret_sharing import ReplicatedSecretSharing
    mat_1 = ReplicatedSecretSharing([e @ b.item[0], e @ b.item[1]], x.party)
    mat_2 = ReplicatedSecretSharing([a.item[0] @ f, a.item[1] @ f], x.party)
    check = -c + mat_1 + mat_2 - e @ f

    x.party.send((x.party.party_id+1) % 3, check.item[0])
    other = x.party.receive((x.party.party_id-1) % 3)
    dif = (other - delta_share).flatten().sum()
    # print("res", res)
    # print("check", check)
    if dif.tensor.item() == 0:
        pass
    return res
