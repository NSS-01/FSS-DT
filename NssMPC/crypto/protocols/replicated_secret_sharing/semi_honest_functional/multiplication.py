import time

from NssMPC import RingTensor
from NssMPC.config import SCALE
from NssMPC.crypto.aux_parameter.truncation_keys.rss_trunc_aux_param import RssTruncAuxParams


def mul_with_out_trunc(x, y):
    z_shared = RingTensor.mul(x.item[0], y.item[0]) + RingTensor.mul(x.item[0], y.item[1]) + RingTensor.mul(x.item[1],
                                                                                                            y.item[0])
    from NssMPC.crypto.primitives.arithmetic_secret_sharing.replicated_secret_sharing import ReplicatedSecretSharing
    result = ReplicatedSecretSharing.reshare(z_shared, x.party)
    return result


def matmul_with_out_trunc(x, y):
    t_i = \
        RingTensor.matmul(x.item[0], y.item[0]) + \
        RingTensor.matmul(x.item[0], y.item[1]) + \
        RingTensor.matmul(x.item[1], y.item[0])
    from NssMPC.crypto.primitives.arithmetic_secret_sharing.replicated_secret_sharing import ReplicatedSecretSharing
    result = ReplicatedSecretSharing.reshare(t_i, x.party)
    return result


def reconstruct3out3(a, party):
    if party.party_id == 0 or party.party_id == 1:
        st = time.time()
        party.send(2, a)
        et = time.time()
        print("Part A OR B sending :", et - st)
    if party.party_id == 2:
        st = time.time()
        a0 = party.receive(0)
        a1 = party.receive(1)
        a_recon = a + a0 + a1
        party.send(0, a_recon)
        party.send(1, a_recon)
        et = time.time()
        print("Part C all :", et - st)
    if party.party_id == 0 or party.party_id == 1:
        st = time.time()
        a_recon = party.receive(2)
        et = time.time()
        print("Part A AND B receive :", et - st)
    return a_recon



def matmul_with_trunc(x, y):
    st = time.time()
    z_shared = RingTensor.matmul(x.item[0], y.item[0]) + RingTensor.matmul(x.item[0], y.item[1]) + RingTensor.matmul(x.item[1],
                                                                                                            y.item[0])
    et = time.time()
    print("                       ")
    print("3-2 -> 3-3 mul time: ", et - st)

    st = time.time()
    r, r_t = x.party.get_param(RssTruncAuxParams, z_shared.numel())
    shape = z_shared.shape
    share = z_shared.flatten()
    # todo：确定用这种方法的话需要改r r_t形式
    r_33 = r.item[0]
    r_t.party = x.party
    # todo:计算原理需要统一
    r_t.dtype = 'float'
    r.dtype = 'float'
    et = time.time()
    print("get trunc aux param time: ", et - st)

    st = time.time()
    delta_share = share - r_33
    delta = reconstruct3out3(delta_share, x.party)
    et = time.time()
    print("reconstruct time: ", et - st)

    st = time.time()
    delta_trunc = delta // SCALE
    et = time.time()
    print("div time: ", et - st)

    st = time.time()
    result = r_t + delta_trunc
    res = result.reshape(shape)
    et = time.time()
    print("reshare time: ", et - st)


    return res