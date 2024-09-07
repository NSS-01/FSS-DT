import math

from config.base_configs import BIT_LEN
from crypto.tensor.ArithmeticSharedRingTensor import ArithmeticSharedRingTensor
from crypto.protocols.arithmetic_secret_sharing.arithmetic_secret_sharing import b2a
from crypto.primitives.function_secret_sharing.dpf import DPF
from crypto.tensor.RingTensor import RingTensor


def equal_match_key_gen_all_dpf(bit_len, num_of_keys, total_len=BIT_LEN):
    total_keys = num_of_keys * total_len // bit_len
    r = RingTensor.random([num_of_keys])
    # r = r.view(-1, total_len // bit_len)
    split_r = split_num(r, bit_len)
    split_r = split_r.reshape(-1)
    r.bit_len = bit_len
    k0, k1 = DPF.gen(total_keys, split_r, RingTensor.convert_to_ring(1))
    r0, r1 = ArithmeticSharedRingTensor.share(r, 2)

    r2 = RingTensor.random([num_of_keys], down_bound=0, upper_bound=total_len // bit_len)
    k20, k21 = DPF.gen(num_of_keys, r2, RingTensor.convert_to_ring(1))
    r20, r21 = ArithmeticSharedRingTensor.share(r2, 2)

    r3 = RingTensor.random([1], down_bound=0, upper_bound=total_len)
    r3.bit_len = int(math.log2(total_len))
    k30, k31 = DPF.gen(1, r3, RingTensor.convert_to_ring(1))
    r30, r31 = ArithmeticSharedRingTensor.share(r3, 2)

    return (k0, r0, k20, r20, k30, r30), (k1, r1, k21, r21, k31, r31)


def equal_match_all_dpf(x: ArithmeticSharedRingTensor, y: ArithmeticSharedRingTensor, key, bit_len):
    k, r, k2, r2, k3, r3 = key
    delta = x - y
    x_shift_shared = delta + ArithmeticSharedRingTensor(r, x.party)
    x_shift = x_shift_shared.restore()

    split_x = split_num(x_shift, bit_len)
    res1 = DPF.eval(split_x, k, x.party.party_id).sum(dim=1)

    res1_shift_shared = r2 + res1
    res1_shift_shared = ArithmeticSharedRingTensor(res1_shift_shared, x.party) - RingTensor.convert_to_ring(
        BIT_LEN // bit_len)
    res1_shift = res1_shift_shared.restore()

    res2 = DPF.eval(res1_shift, k2, x.party.party_id)
    res2 = res2.sum().reshape(1, -1)

    res2_shift_shared = r3 + res2
    res2_shift_shared = ArithmeticSharedRingTensor(res2_shift_shared, x.party) - RingTensor.convert_to_ring(1)
    res2_shift = res2_shift_shared.restore()
    res2_shift.bit_len = int(math.log2(BIT_LEN))

    res = DPF.eval(res2_shift, k3, x.party.party_id)

    return ArithmeticSharedRingTensor(RingTensor(res), x.party)


# TODO: 第二轮的处理根据数据的不同范围确定不同的环的大小
def equal_match_key_gen_one_msb(num_of_keys, total_len=BIT_LEN):
    r2 = RingTensor.random([num_of_keys], down_bound=0, upper_bound=total_len)
    k20, k21 = DPF.gen(num_of_keys, r2, RingTensor.convert_to_ring(1))
    r20, r21 = ArithmeticSharedRingTensor.share(r2, 2)

    r3 = RingTensor.random([1], down_bound=0, upper_bound=total_len)
    r3.bit_len = int(math.log2(total_len))
    k30, k31 = DPF.gen(1, r3, RingTensor.convert_to_ring(1))
    r30, r31 = ArithmeticSharedRingTensor.share(r3, 2)

    return (k20, r20, k30, r30), (k21, r21, k31, r31)


def equal_match_one_msb(x: ArithmeticSharedRingTensor, y: ArithmeticSharedRingTensor, mask: ArithmeticSharedRingTensor, key):
    k2, r2, k3, r3 = key
    x_bit = to_bin(x.ring_tensor.reshape(1, -1))
    y_bit = to_bin(y.ring_tensor.reshape(1, -1))

    delta = x_bit ^ y_bit

    print(delta.shape)

    delta = b2a(delta.tensor, x.party)
    delta = delta.reshape(-1, BIT_LEN)

    delta = RingTensor(delta.ring_tensor.tensor, x.ring_tensor.dtype, x.ring_tensor.scale)
    res1 = delta.sum(1)

    # res1 = res1 * mask

    res1_shift_shared = r2 + res1
    res1_shift_shared = ArithmeticSharedRingTensor(res1_shift_shared, x.party)
    res1_shift = res1_shift_shared.restore()

    res2 = DPF.eval(res1_shift, k2, x.party.party_id)
    res2 = res2.sum().reshape(1, -1)

    res2_shift_shared = r3 + res2
    res2_shift_shared = ArithmeticSharedRingTensor(res2_shift_shared, x.party) - RingTensor.convert_to_ring(1)
    res2_shift = res2_shift_shared.restore()
    res2_shift.bit_len = int(math.log2(BIT_LEN))

    res = DPF.eval(res2_shift, k3, x.party.party_id)

    return ArithmeticSharedRingTensor(RingTensor(res), x.party)


def split_num(x, bit_len):
    x = x.reshape(-1, 1)
    small_x = x % (2 ** bit_len)
    x_ = x >> bit_len
    num = x.bit_len // bit_len
    for _ in range(num - 1):
        small_x = small_x.cat(x_ % (2 ** bit_len), dim=1)
        x_ = x_ >> bit_len
    small_x.bit_len = bit_len
    return small_x


def to_bin(x: RingTensor):
    mask = 1
    shifted = x.unsqueeze(-1) >> x.arange(BIT_LEN - 1, -1, -1)
    # 使用按位与操作获取每个位置的二进制位
    binary_matrix = (shifted & mask).squeeze(1)
    return binary_matrix
