from crypto.primitives.auxiliary_parameter.parameter import Parameter
from crypto.tensor.ArithmeticSharedRingTensor import ArithmeticSharedRingTensor
from crypto.tensor.RingTensor import RingTensor


class SelectLinGeLU(object):
    @staticmethod
    def eval(x, key, w, d):
        return gelu_select_lin_eval(key, w, d, x)


class SelectLinGeLUKey(Parameter):
    def __init__(self):
        self.p = None
        self.q = None
        self.w = None
        self.d = None

    @staticmethod
    def gen(num_of_keys, sigma_r, table_scale_bit=6):
        return gelu_select_lin_gen(num_of_keys, sigma_r, table_scale_bit)


def gelu_select_lin_gen(num_of_keys, sigma_r, table_scale_bit=6):
    w = RingTensor.random([num_of_keys], down_bound=0, upper_bound=4)
    d = RingTensor.random([num_of_keys], down_bound=0, upper_bound=4)
    i = (w * 2 + d) % 4

    p = RingTensor([0, 0, -1, 1]).repeat(num_of_keys, 1)
    p = RingTensor.row_shift(p, i.tolist())

    q = RingTensor([2 ** (table_scale_bit + 2) - 1, 2 ** (table_scale_bit + 2) - 1]).repeat(num_of_keys, 1)
    q = RingTensor.cat((q, sigma_r.view(-1, 1), (-sigma_r).view(-1, 1)), dim=1)
    q = RingTensor.row_shift(q, i.tolist())

    k0 = SelectLinGeLUKey()
    k1 = SelectLinGeLUKey()

    k0.p, k1.p = ArithmeticSharedRingTensor.share(p, 2)
    k0.q, k1.q = ArithmeticSharedRingTensor.share(q, 2)

    r = RingTensor.random([num_of_keys], down_bound=0, upper_bound=4)
    k0.w = r
    k1.w = (w - r) % 4

    r = RingTensor.random([num_of_keys], down_bound=0, upper_bound=4)
    k0.d = r
    k1.d = (d - r) % 4

    return k0, k1


def gelu_select_lin_eval(key, w, d, x_shift: RingTensor):
    shape = x_shift.shape

    w_shift = ArithmeticSharedRingTensor((key.w + w.flatten()), w.party)
    d_shift = ArithmeticSharedRingTensor((key.d + d.flatten()), d.party)

    w_shift = w_shift.restore()
    d_shift = d_shift.restore()

    i_shift = (w_shift * 2 + d_shift) % 4

    key.p.dtype = x_shift.dtype
    key.q.dtype = x_shift.dtype

    key.p = key.p * x_shift.scale

    return (key.p[i_shift.tensor] * x_shift.flatten() + key.q[i_shift.tensor]).reshape(shape)
