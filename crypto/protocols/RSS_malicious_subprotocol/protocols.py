import time

from common.utils.tensor_utils import list_rotate
from crypto.tensor.ArithmeticSharedRingTensor import ArithmeticSharedRingTensor
from config.base_configs import *
from crypto.primitives.arithmetic_secret_sharing.replicated_secret_sharing import ReplicatedSecretSharing
from crypto.primitives.function_secret_sharing.verifiable_dpf import VerifiableDPFKey
from crypto.protocols.comparison.verifiable_sigma import verifiable_sigma_gen, verifiable_sigma_eval
from crypto.tensor.RingTensor import RingTensor


def rand_with_prg(num_of_value, party):
    """
    Generating an RSS sharing <c> for a random value c
    :param num_of_value: the number of random value to generate
    :param party: party instance
    :return: RSS sharing <c>
    """
    # for i
    r_0 = party.prg_0.random(num_of_value)
    # for i+1
    r_1 = party.prg_1.random(num_of_value)
    # 由于我们直接从环上生成的随机数，因此不需要convert
    r_0_ring = RingTensor(r_0, party.dtype, party.scale)
    r_1_ring = RingTensor(r_1, party.dtype, party.scale)
    r = ReplicatedSecretSharing([r_0_ring, r_1_ring], party)
    return r


def rand_like(x, party):
    r = rand_with_prg(x.numel(), party)
    r = r.reshape(x.shape)
    return r


def open(x):
    """
    Open an RSS sharing <x> to each party

    :param x: an RSS sharing <x>
    :return: the restore value of <x>
    """

    # send x0 to P_{i+1}
    x.party.send_ring_tensor_to((x.party.party_id + 1) % 3, x.replicated_shared_tensor[0])
    # receive x2 from P_{i+1}
    x_2_0 = x.party.receive_ring_tensor_from((x.party.party_id + 2) % 3)
    # send x1 to P_{i-1}
    x.party.send_ring_tensor_to((x.party.party_id + 2) % 3, x.replicated_shared_tensor[1])
    # receive x2 from P_{i-1}
    x_2_1 = x.party.receive_ring_tensor_from((x.party.party_id + 1) % 3)

    cmp = x_2_0 - x_2_1
    # print(cmp)
    cmp = cmp.tensor.flatten().sum(axis=0)
    # if cmp != 0:
    #     raise ValueError("The two parties' calculations do not agree, and there may be a malicious party involved!")

    return x.replicated_shared_tensor[0] + x.replicated_shared_tensor[1] + x_2_0


def coin(num_of_value, party):
    """
    Outputs a random value r to all the parties
    :param num_of_value: the number of random value to generate
    :param party: party instance
    :return: random value r
    """
    rss_r = rand_with_prg(num_of_value, party)
    r = open(rss_r)
    return r


def recon(x, party_id):
    """
    Reconstructs a consistent RSS sharing ⟨x⟩ to P_i
    :param x: an RSS sharing ⟨x⟩
    :param party_id: the party id of P_i
    :return: plain text x only known to P_i
    """
    # P_{i+1} send x_{i+2} to P_{i}
    if x.party.party_id == (party_id + 1) % 3:
        x.party.send_ring_tensor_to(party_id, x.replicated_shared_tensor[1])
    # P_{i-1} send x_{i+2} to P_{i}
    elif x.party.party_id == (party_id + 2) % 3:
        x.party.send_ring_tensor_to(party_id, x.replicated_shared_tensor[0])

    elif x.party.party_id == party_id:
        x_2_0 = x.party.receive_ring_tensor_from((x.party.party_id + 2) % 3)
        x_2_1 = x.party.receive_ring_tensor_from((x.party.party_id + 1) % 3)

        cmp = x_2_0 - x_2_1
        cmp = cmp.tensor.flatten().sum(axis=0)
        # if cmp != 0:
        #     raise ValueError("The two parties' calculations do not agree, and there may be a malicious party involved!")
        if x.party.party_id == party_id:
            return x.replicated_shared_tensor[0] + x.replicated_shared_tensor[1] + x_2_0
    return None


def receive_share_from(input_id, party):
    """
    Receives a shared value from P_i
    :param party: party instance
    :param input_id: the id of input party
    :return: an RSS sharing ⟨x⟩
    """
    # receive shape from P_i
    shape_of_received_shares = party.receive_torch_tensor_from(input_id)
    tmp = torch.zeros(shape_of_received_shares.tolist())
    r = rand_like(tmp, party)
    r_recon = recon(r, input_id)

    # receive delta from P_{input_id}
    delta = party.receive_ring_tensor_from(input_id)

    # check if delta is same
    other_id = (0 + 1 + 2) - party.party_id - input_id
    # send delta to P_{other_id}
    party.send_ring_tensor_to(other_id, delta)
    delta_other = party.receive_ring_tensor_from(other_id)

    cmp = delta - delta_other
    cmp = cmp.tensor.flatten().sum(axis=0)
    if cmp != 0:
        raise ValueError("The two parties' calculations do not agree, and there may be a malicious party involved!")
    res = r + delta
    return res


def share(x, party):
    """
    Shares a secret x from P_i among three parties.
    :param x: public input x only input by P_i
    :param party: party instance
    :return: an RSS sharing ⟨x⟩
    """
    if not isinstance(x, RingTensor):
        raise TypeError("unsupported data type(s) ")

    # todo:这里发送x的形状是否合适？
    shape = x.shape
    # send shape to P_{i+1} and P_{i-1}
    party.send_torch_tensor_to((party.party_id + 1) % 3, torch.tensor(shape))
    party.send_torch_tensor_to((party.party_id + 2) % 3, torch.tensor(shape))

    r = rand_like(x, party)
    r_recon = recon(r, party.party_id)
    delta = x - r_recon
    # broadcasts delta to all parties
    party.send_ring_tensor_to((party.party_id + 1) % 3, delta)
    party.send_ring_tensor_to((party.party_id + 2) % 3, delta)
    res = r + delta
    return res


def mul_semi_honest(x, y):
    """
    Multiplication of two RSS sharings ⟨x⟩ and ⟨y⟩
    :param x: an RSS sharing ⟨x⟩
    :param y: an RSS sharing ⟨y⟩
    :return: an RSS sharing ⟨x*y⟩
    """

    t_i = \
        x.replicated_shared_tensor[0].tensor * y.replicated_shared_tensor[0].tensor + \
        x.replicated_shared_tensor[0].tensor * y.replicated_shared_tensor[1].tensor + \
        x.replicated_shared_tensor[1].tensor * y.replicated_shared_tensor[0].tensor
    t_i = RingTensor(t_i, x.replicated_shared_tensor[0].dtype,
                     x.replicated_shared_tensor[0].scale, x.device)

    r = rand_like(x, x.party)

    z_i = t_i + r.replicated_shared_tensor[0] - r.replicated_shared_tensor[1]

    # send z_i to P_{i-1}
    x.party.send_ring_tensor_to((x.party.party_id + 2) % 3, z_i)
    # receive z_{i+1} from P_{i-1}
    z_i_1 = x.party.receive_ring_tensor_from((x.party.party_id + 1) % 3)

    res = ReplicatedSecretSharing([z_i, z_i_1], x.party)

    return res


def mul(x: ReplicatedSecretSharing, y: ReplicatedSecretSharing):
    """
    Multiplication of two RSS sharings ⟨x⟩ and ⟨y⟩
    :param x: an RSS sharing ⟨x⟩
    :param y: an RSS sharing ⟨y⟩
    :return: an RSS sharing ⟨x*y⟩
    """
    res = mul_semi_honest(x, y)

    a, b, c = x.party.beaver_provider.get_triples(res.shape)
    e = x + a
    f = y + b
    e = open(e)
    f = open(f)

    check_zero(res - c + b * e + a * f - e * f)

    return res


def matmul(x: ReplicatedSecretSharing, y: ReplicatedSecretSharing):
    """
    Matrix Multiplication of two RSS sharings ⟨x⟩ and ⟨y⟩
    :param x: an RSS sharing ⟨x⟩
    :param y: an RSS sharing ⟨y⟩
    :return: an RSS sharing ⟨x@y⟩
    """
    t_i = \
        RingTensor.matmul(x.replicated_shared_tensor[0], y.replicated_shared_tensor[0]) + \
        RingTensor.matmul(x.replicated_shared_tensor[0], y.replicated_shared_tensor[1]) + \
        RingTensor.matmul(x.replicated_shared_tensor[1], y.replicated_shared_tensor[0])

    r = rand_like(t_i, x.party)

    z_i = t_i + r.replicated_shared_tensor[0] - r.replicated_shared_tensor[1]

    # send z_i to P_{i-1}
    x.party.send_ring_tensor_to((x.party.party_id + 2) % 3, z_i)
    # receive z_{i+1} from P_{i-1}
    z_i_1 = x.party.receive_ring_tensor_from((x.party.party_id + 1) % 3)

    res = ReplicatedSecretSharing([z_i, z_i_1], x.party)

    if x.replicated_shared_tensor[0].dtype == "float":
        res = truncate(res)

    a = ReplicatedSecretSharing.zeros_like(x, x.party)
    b = ReplicatedSecretSharing.zeros_like(y, x.party)
    c = ReplicatedSecretSharing.zeros_like(res, x.party)
    e = x + a
    f = y + b
    e = open(e)
    f = open(f)

    mat_1 = ReplicatedSecretSharing([e @ b.replicated_shared_tensor[0], e @ b.replicated_shared_tensor[1]], x.party)
    mat_2 = ReplicatedSecretSharing([a.replicated_shared_tensor[0] @ f, a.replicated_shared_tensor[1] @ f], x.party)

    check_zero(res - c + mat_1 + mat_2 - e @ f)

    return res


def mul_with_mac_check(x: ReplicatedSecretSharing, y: ReplicatedSecretSharing, mac_x: ReplicatedSecretSharing,
                       mac_y: ReplicatedSecretSharing, mac_key):
    res = mul_semi_honest(x, y)
    mac_res = mul_semi_honest(mac_x, mac_y)

    mac_check(res, mac_res, mac_key)

    return res


def mul_2pc(x: RingTensor, y: RingTensor, party, other_party_id):
    # a, b, c, _, _, _, _ = party.beaver_provider_2pc.get_triples(x.shape)

    a = RingTensor.convert_to_ring(1)
    b = RingTensor.convert_to_ring(1)
    c = RingTensor.convert_to_ring(2)

    e = x - a
    f = y - b

    e_and_f = e.cat(f)
    party.send_ring_tensor_to(other_party_id, e_and_f)
    other_e_and_f = party.receive_ring_tensor_from(other_party_id)
    length = other_e_and_f.shape[0]
    other_e = e_and_f[:length // 2]
    other_f = e_and_f[length // 2:]

    common_e = e + other_e
    common_f = f + other_f

    i = party.party_id % 2
    if party.party_id == 2:
        i = (i + other_party_id + 1) % 2

    res1 = common_e.tensor * common_f.tensor * i
    res2 = a.tensor * common_f.tensor
    res3 = common_e.tensor * b.tensor
    res = res1 + res2 + res3 + c.tensor

    res = RingTensor(res, dtype=x.dtype)

    return res


def mul_2pc_with_mac_check(x: RingTensor, y: RingTensor, party, other_party_id):
    a, b, c, mac_key, mac_a, mac_b, mac_c = party.beaver_provider_2pc.get_triples(x.shape)

    a = RingTensor.convert_to_ring(1)
    b = RingTensor.convert_to_ring(1)
    c = RingTensor.convert_to_ring(2)

    e = x - a
    f = y - b

    e_and_f = e.cat(f)
    party.send_ring_tensor_to(other_party_id, e_and_f)
    other_e_and_f = party.receive_ring_tensor_from(other_party_id)
    length = other_e_and_f.shape[0]
    other_e = other_e_and_f[:length // 2]
    other_f = other_e_and_f[length // 2:]

    common_e = e + other_e
    common_f = f + other_f

    i = party.party_id % 2
    if party.party_id == 2:
        i = (i + other_party_id + 1) % 2

    res = common_e.tensor * common_f.tensor * i + a.tensor * common_f.tensor + common_e.tensor * b.tensor + c.tensor

    mac_res = mac_key.tensor * common_e.tensor * common_f.tensor * i \
              + mac_a.tensor * common_f.tensor \
              + common_e.tensor * mac_b.tensor + mac_c.tensor

    res = RingTensor(res, dtype=x.dtype)
    mac_res = RingTensor(mac_res, dtype=x.dtype)

    mac_check_2pc(res, mac_res, mac_key, party, other_party_id)
    return res


def mac_check_2pc(x: RingTensor, mac_x: RingTensor, mac_key, party, other_party_id):
    if DEBUG:
        mac_key = RingTensor(torch.full(x.shape, mac_key.tensor), dtype=x.dtype, device=x.device)
    x_mac = mul_2pc(x, mac_key, party, other_party_id)
    delta = x_mac - mac_x
    # TODO: mac_check应该需要一轮通信，在这模拟一下，这么做好像不对而且可能有安全问题
    party.send_ring_tensor_to(other_party_id, delta)
    delta_other = party.receive_ring_tensor_from(other_party_id)

    delta += delta_other

    res = (delta.tensor == 0) + 0
    # if res.flatten().sum() != delta.numel():
    #     raise ValueError("The two parties' calculations do not agree, and there may be a malicious party involved!")


def check_zero(x):
    """
    Checks if an RSS sharing ⟨x⟩ is zero
    :param x: an RSS sharing ⟨x⟩
    :return: 1 if x is zero, 0 otherwise
    """
    r = rand_like(x, x.party)
    w = mul_semi_honest(x, r)
    w_open = open(w)
    res = (w_open.tensor == 0) + 0
    # if res.flatten().sum() != x.numel():
    #     raise ValueError("The two parties' calculations do not agree, and there may be a malicious party involved!")
    return res


def check_is_all_element_equal(x, y):
    """
    Checks if all elements of x are equal to all elements of y
    :param x: a tensor x
    :param y: a tensor y
    :return: 1 if all elements of x are equal to all elements of y, 0 otherwise
    """
    cmp = x - y
    cmp = cmp.tensor.flatten().sum(axis=0)
    if cmp != 0:
        raise ValueError("The two parties' calculations do not agree, and there may be a malicious party involved!")
    return 1


def bit_injection(x):
    """
    Bit injection of an RSS sharing ⟨x⟩B
    :param x: an RSS binary sharing ⟨x⟩B
    :return: an RSS arithmetic sharing ⟨x⟩A
    """
    x1 = x.replicated_shared_tensor[0]
    x2 = x.replicated_shared_tensor[1]

    # 构建x1,x2,x3的加法秘密分享
    zeros = RingTensor.zeros_like(x1)
    if x.party.party_id == 0:
        a_x1 = ReplicatedSecretSharing([x1, zeros], x.party)
        a_x2 = ReplicatedSecretSharing([zeros, x2], x.party)
        a_x3 = ReplicatedSecretSharing([zeros, zeros], x.party)
    elif x.party.party_id == 1:
        a_x1 = ReplicatedSecretSharing([zeros, zeros], x.party)
        a_x2 = ReplicatedSecretSharing([x1, zeros], x.party)
        a_x3 = ReplicatedSecretSharing([zeros, x2], x.party)
    else:
        a_x1 = ReplicatedSecretSharing([zeros, x2], x.party)
        a_x2 = ReplicatedSecretSharing([zeros, zeros], x.party)
        a_x3 = ReplicatedSecretSharing([x1, zeros], x.party)

    mul1 = mul(a_x1, a_x2)
    d = a_x1 + a_x2 - mul1 - mul1

    mul2 = mul(d, a_x3)
    b = d + a_x3 - mul2 - mul2

    return b


def b2a_2pc(x: RingTensor, party, party_id_0, party_id_1):
    zero = RingTensor.zeros_like(x)
    if party.party_id == party_id_0:
        a = x
        b = zero
        other_party = party_id_1
    else:
        a = zero
        b = x
        other_party = party_id_0
    return a + b - mul_2pc_with_mac_check(a, b, party, other_party) * 2


def get_msb(x: ReplicatedSecretSharing):
    return get_msb_malicious(x)


def get_msb_malicious(x: ReplicatedSecretSharing):
    """
    Gets the most significant bit of an RSS sharing ⟨x⟩
    :param x: an RSS sharing ⟨x⟩
    :return: the most significant bit of x
    """
    party = x.party
    key_from_1, key_from_2 = party.key_provider.get_parameters(x.numel(), 3)
    self_key0, self_key1 = party.key_provider.get_self_keys_by_pointer(x.numel())
    r_in0 = self_key0.r_in
    r_in1 = self_key1.r_in

    rin_from_p1 = key_from_1.r_in
    rin_from_p2 = key_from_2.r_in

    p1k1 = key_from_2.ver_dpf_key
    p2k0 = key_from_1.ver_dpf_key
    c_from_p1 = key_from_1.c
    c_from_p2 = key_from_2.c

    if DEBUG:
        r_list = [ReplicatedSecretSharing([r_in0, r_in1], party),
                  ReplicatedSecretSharing([RingTensor.convert_to_ring(0), rin_from_p1], party),
                  ReplicatedSecretSharing([rin_from_p2, RingTensor.convert_to_ring(0)], party)]
    else:
        r_list = [ReplicatedSecretSharing([r_in0, r_in1], party).reshape(x.shape),
                  ReplicatedSecretSharing([r_in1, rin_from_p1], party).reshape(x.shape),
                  ReplicatedSecretSharing([rin_from_p2, r_in0], party).reshape(x.shape)]

    r_list = list_rotate(r_list, party.party_id)

    r0 = x + r_list[0]
    r1 = x + r_list[1]
    r2 = x + r_list[2]

    rb1_list = []
    rb2_list = []

    rb1_list.append(recon(r1, 0))
    rb2_list.append(recon(r2, 0))

    rb1_list.append(recon(r2, 1))
    rb2_list.append(recon(r0, 1))

    rb1_list.append(recon(r0, 2))
    rb2_list.append(recon(r1, 2))

    rb1 = rb1_list[party.party_id].to(x.device)
    rb2 = rb2_list[party.party_id].to(x.device)

    out_1, pi1 = verifiable_sigma_eval(1, (p1k1.to(DEVICE), c_from_p1), rb1)
    out_2, pi2 = verifiable_sigma_eval(0, (p2k0.to(DEVICE), c_from_p2), rb2)

    if party.party_id == 2:
        party.send_ring_tensor_to((party.party_id + 1) % 3, pi2)
        o_pi2 = party.receive_ring_tensor_from((party.party_id + 1) % 3)
        check_is_all_element_equal(pi2, o_pi2)

        party.send_ring_tensor_to((party.party_id + 2) % 3, pi1)
        o_pi1 = party.receive_ring_tensor_from((party.party_id + 2) % 3)
        check_is_all_element_equal(pi1, o_pi1)
    else:
        party.send_ring_tensor_to((party.party_id + 2) % 3, pi1)
        o_pi1 = party.receive_ring_tensor_from((party.party_id + 2) % 3)
        check_is_all_element_equal(pi1, o_pi1)

        party.send_ring_tensor_to((party.party_id + 1) % 3, pi2)
        o_pi2 = party.receive_ring_tensor_from((party.party_id + 1) % 3)
        check_is_all_element_equal(pi2, o_pi2)

    out = out_1 ^ out_2

    party.send_ring_tensor_to((party.party_id + 1) % 3, out)
    out_from_2 = party.receive_ring_tensor_from((party.party_id + 2) % 3)

    rss_out = ReplicatedSecretSharing([out_from_2, out], party)
    return bit_injection(rss_out)


def get_msb_one_semi_honest(x: ReplicatedSecretSharing):
    """
    假设Party 0是半诚实可信的
    :param x:
    :return:
    """
    party = x.party
    if party.party_id == 0:
        key0, key1 = party.key_provider.get_parameters(x.numel(), 3)

        k0, c0, r_in0 = key0.ver_dpf_key, key0.c, key0.r_in
        k1, c1, r_in1 = key1.ver_dpf_key, key1.c, key1.r_in

        party.send_params_to(1, k0)
        party.send_params_to(2, k1)

        party.send_ring_tensor_to(1, c0)
        party.send_ring_tensor_to(2, c1)

        party.send_ring_tensor_to(1, r_in1)
        party.send_ring_tensor_to(2, r_in0)

        party.receive_torch_tensor_from(1)
        party.receive_torch_tensor_from(2)

        r = x + ReplicatedSecretSharing([r_in0, r_in1], party)

        recon(r, 1)
        recon(r, 2)

        out = RingTensor.zeros(x.shape, x.replicated_shared_tensor[0].dtype, x.replicated_shared_tensor[0].scale)

    else:
        k = VerifiableDPFKey.dic_to_key(party.receive_params_dict_from(0))
        c = party.receive_ring_tensor_from(0)
        r_in = party.receive_ring_tensor_from(0)

        party.send_torch_tensor_to(0, torch.tensor(1))

        if DEBUG:
            rss_in = [r_in, RingTensor.convert_to_ring(0)]
        else:
            rss_in = [r_in.reshape(x.shape), RingTensor.convert_to_ring(0)]

        rss_in = list_rotate(rss_in, party.party_id - 1)

        r = x + ReplicatedSecretSharing(rss_in, party)

        r_list = [recon(r, 1), recon(r, 2)]

        x_shift = r_list[party.party_id - 1].to(x.device)

        out, pi = verifiable_sigma_eval(party.party_id - 1, (k, c), x_shift)

        party.send_ring_tensor_to(party.party_id % 2 + 1, pi)
        o_pi = party.receive_ring_tensor_from(party.party_id % 2 + 1)
        check_is_all_element_equal(pi, o_pi)

    ran = rand_like(out, party)
    out = out ^ ran.replicated_shared_tensor[0].get_bit(1) ^ ran.replicated_shared_tensor[1].get_bit(1)

    party.send_ring_tensor_to((party.party_id + 1) % 3, out)
    out_from_2 = party.receive_ring_tensor_from((party.party_id + 2) % 3)

    rss_out = ReplicatedSecretSharing([out_from_2, out], party)
    return bit_injection(rss_out)


def mac_check(x: ReplicatedSecretSharing, mx: ReplicatedSecretSharing, mac_key: ReplicatedSecretSharing):
    r = rand_like(x, x.party)
    mr = mul_semi_honest(r, mac_key)
    ro = coin(x.numel(), x.party).reshape(x.shape)
    v = r + x * ro
    w = mr + mx * ro
    v = open(v)
    check_zero(w - mac_key * v)


def truncate(share: ReplicatedSecretSharing, scale=SCALE):
    # TODO: truncate有问题，在此只是模拟通信
    # r, r_t = truncate_preprocess(share, scale)
    # delta_share = share - r
    delta = open(share)
    return share

    # return r_t + delta // scale


def truncate_preprocess(share: ReplicatedSecretSharing, scale=SCALE):
    # TODO: 后续是否要改成离线操作
    r_share = rand_like(share, share.party)
    r_share = ReplicatedSecretSharing(
        [r_share.replicated_shared_tensor[0] // 3, r_share.replicated_shared_tensor[1] // 3],
        share.party)  # TODO: 这样处理不好
    r_t_share = ReplicatedSecretSharing(
        [r_share.replicated_shared_tensor[0] // scale, r_share.replicated_shared_tensor[1] // scale], share.party)

    r_2 = rand_like(share, share.party)
    r_3 = rand_like(share, share.party)

    r_t_2 = rand_like(share, share.party)
    r_t_2 = ReplicatedSecretSharing(
        [r_t_2.replicated_shared_tensor[0] // scale, r_t_2.replicated_shared_tensor[1] // scale], share.party)
    r_t_3 = rand_like(share, share.party)
    r_t_3 = ReplicatedSecretSharing(
        [r_t_3.replicated_shared_tensor[0] // scale, r_t_3.replicated_shared_tensor[1] // scale], share.party)

    r_1 = r_share - r_2 - r_3
    r_t_1 = r_t_share - r_t_2 - r_t_3

    r_0_list = []
    r_1_list = []
    r_t_0_list = []
    r_t_1_list = []

    r_0_list.append(recon(r_1, 0))
    r_1_list.append(recon(r_2, 0))

    r_t_0_list.append(recon(r_t_1, 0))
    r_t_1_list.append(recon(r_t_2, 0))

    r_0_list.append(recon(r_2, 1))
    r_1_list.append(recon(r_3, 1))

    r_t_0_list.append(recon(r_t_2, 1))
    r_t_1_list.append(recon(r_t_3, 1))

    r_0_list.append(recon(r_3, 2))
    r_1_list.append(recon(r_1, 2))

    r_t_0_list.append(recon(r_t_3, 2))
    r_t_1_list.append(recon(r_t_1, 2))

    r = ReplicatedSecretSharing([r_0_list[share.party.party_id], r_1_list[share.party.party_id]], share.party)
    r_t = ReplicatedSecretSharing([r_t_0_list[share.party.party_id], r_t_1_list[share.party.party_id]], share.party)

    return r, r_t
