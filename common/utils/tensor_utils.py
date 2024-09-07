import torch

from config.base_configs import data_type


def fast_concat(tensors, dim=0):
    shape = list(tensors[0].shape)
    shape[dim] = sum([t.shape[dim] for t in tensors])
    result = torch.empty(*shape, dtype=tensors[0].dtype, device=tensors[0].device)
    offset = 0
    for t in tensors:
        size = t.shape[dim]
        result.narrow(dim, offset, size).copy_(t)
        offset += size
    return result


def int64_to_int32x2(tensor_64: torch.Tensor):
    if tensor_64.dtype != torch.int64:
        raise ValueError("Input tensor must be of type torch.int64")

    high_32 = (tensor_64 >> 32).to(torch.int32)
    low_32 = (tensor_64 & 0xFFFFFFFF).to(torch.int32)

    out_shape = list(tensor_64.shape)
    out_shape[-1] *= 2
    out = torch.empty(out_shape, dtype=torch.int32)

    out[..., 0::2] = high_32
    out[..., 1::2] = low_32

    return out


def list_rotate(list_before, n):
    """
    Rotate a list by n steps
    对于list，转成tensor然后用torch.roll再转回来好像更快
    :param list_before:
    :param n:
    :return:
    """
    list_after = []
    for i in range(len(list_before)):
        list_after.append(list_before[(i - n) % len(list_before)])

    return list_after


def rows_shift(inputs: torch.Tensor, offsets):
    """
    :param inputs: 待偏移张量，目前必须是两维
    :param offsets: 偏移量，正数左移，负数右移
    :return:
    """
    if isinstance(offsets, list):
        offsets = torch.tensor(offsets, dtype=data_type, device=inputs.device)

    n = inputs.shape[1]
    rows = torch.arange(inputs.shape[0]).view(-1, 1)  # 每一行的行号
    indices = (torch.arange(n, device=offsets.device) + offsets.view(-1, 1)) % n  # 对每个偏移量x生成索引并应用模n
    result = inputs[rows, indices]
    return result
