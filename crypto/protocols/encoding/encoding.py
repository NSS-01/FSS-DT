import torch

from crypto.tensor.RingTensor import RingTensor


def zero_encoding(x: RingTensor):
    bit_len = x.bit_len
    zero_encoding_list = RingTensor.empty([bit_len], dtype=x.dtype , device=x.device, scale=x.scale)
    for i in range(bit_len):
        current_bit = x.get_bit(i)
        if current_bit == 0:
            current_encoding = x.bit_slice(i + 1, bit_len) << 1 | 1
            zero_encoding_list[i] = current_encoding
        else:
            current_encoding = RingTensor.random(x.shape)
            zero_encoding_list[i] =  current_encoding
    fake_mask = RingTensor.empty([bit_len], dtype=x.dtype , device=x.device, scale=x.scale)
    for i in range(bit_len):
        fake_mask[i] = RingTensor(1 - x.get_bit(i))
    return zero_encoding_list, fake_mask


def one_encoding(x:RingTensor):
    bit_len = x.bit_len
    one_encoding_list = RingTensor.empty([bit_len], dtype=x.dtype , device=x.device, scale=x.scale)
    for i in range(bit_len):
        current_bit = x.get_bit(i)
        if current_bit == 1:
            current_encoding = x.bit_slice(i, bit_len)
            one_encoding_list[i] = current_encoding
        else:
            current_encoding = RingTensor.random(x.shape)
            one_encoding_list[i] = current_encoding
    fake_mask = RingTensor.empty([bit_len], dtype=x.dtype , device=x.device, scale=x.scale)
    for i in range(bit_len):
        fake_mask[i] = RingTensor(x.get_bit(i))
    return one_encoding_list, fake_mask



# x = RingTensor.convert_to_ring(6)
# x.bit_len = 4
# y = RingTensor.convert_to_ring(10)
# y.bit_len = 4
# #
# x_encoding, x_mask = zero_encoding(x)
# print(x_encoding)
# # for encoding in x_encoding:
# #     print(bin(encoding.tensor))
# print("x_mask", x_mask)
# #
#
# y_encoding, y_mask = one_encoding(y)
# print("y_encoding", y_encoding)
# # for encoding in y_encoding:
# #     print(bin(encoding.tensor))
# print("y_mask", y_mask)
#
