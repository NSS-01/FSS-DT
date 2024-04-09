from config.base_configs import *
from crypto.primitives.auxiliary_parameter.parameter import Parameter
from crypto.tensor.RingTensor import RingTensor


class Wrap(Parameter):
    def __init__(self, r=None, theta_r=None):
        self.r = r
        self.theta_r = theta_r

    @staticmethod
    def gen(num_of_params):
        from crypto.tensor.ArithmeticSharedRingTensor import ArithmeticSharedRingTensor
        r = RingTensor.random([num_of_params])
        r0, r1 = ArithmeticSharedRingTensor.share(r, 2)
        theta_r = count_wraps([r0.tensor, r1.tensor])

        theta_r0, theta_r1 = ArithmeticSharedRingTensor.share(RingTensor(theta_r), 2)

        wrap_0 = Wrap(r0.tensor, theta_r0.tensor)
        wrap_1 = Wrap(r1.tensor, theta_r1.tensor)

        return wrap_0, wrap_1


def truncate(share, scale):
    """
    基于CrypTen的截断方法
    :param share: ASS乘法得到的未经截断的原始结果
    :param scale: 截断位数
    :return: 截断后的的乘法结果
    """
    wrap_count = wraps(share)
    share_tensor = share.tensor
    share_tensor = share_tensor.div_(scale, rounding_mode="trunc").to(data_type)
    share_tensor -= wrap_count * (((RING_MAX // 4) // scale) * 4)
    return share.__class__(RingTensor(share_tensor, dtype=share.dtype), share.party)


def wraps(share):
    """
    Privately computes the number of wraparounds for a set a shares
    To do so, we note that:
        [theta_x] = theta_z + [beta_xr] - [theta_r] - [eta_xr]

    Where [theta_i] is the wraps for a variable i
          [beta_ij] is the differential wraps for variables i and j
          [eta_ij]  is the plaintext wraps for variables i and j

    Note: Since [eta_xr] = 0 with probability 1 - |x| / Q for modulus Q, we
    can make the assumption that [eta_xr] = 0 with high probability.

    两方情况
    """

    share_tensor = share.tensor
    wrap = share.party.wrap_provider.get_parameters(share_tensor.numel())
    r, theta_r = wrap.r, wrap.theta_r
    if not DEBUG:
        r = r.reshape(share_tensor.shape)
        theta_r = r.reshape(share_tensor.shape)

    beta_xr = count_wraps([share_tensor, r])
    z = share_tensor + r

    if share.party.party_id == 0:
        share.party.send(z)
        return beta_xr - theta_r

    if share.party.party_id == 1:
        z_other = share.party.receive()
        theta_z = count_wraps([z_other, z])

        return theta_z + beta_xr - theta_r


def count_wraps(share_list):
    """Computes the number of overflows or underflows in a set of shares

    We compute this by counting the number of overflows and underflows as we
    traverse the list of shares.
    """
    result = torch.zeros_like(share_list[0], dtype=data_type)
    prev = share_list[0]
    for cur in share_list[1:]:
        next = cur + prev
        result -= ((prev < 0) & (cur < 0) & (next > 0)).to(data_type)  # underflow
        result += ((prev > 0) & (cur > 0) & (next < 0)).to(data_type)  # overflow
        prev = next
    return result
