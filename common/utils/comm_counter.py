import torch


def count_bytes(a):
    from crypto.tensor.RingTensor import RingTensor
    # TODO: 字典怎么处理
    if isinstance(a, torch.Tensor):
        return a.element_size() * a.nelement()
    elif isinstance(a, RingTensor):
        return count_bytes(a.tensor)
    elif isinstance(a, dict):
        return 0
    elif isinstance(a, int):
        return 4
    elif isinstance(a, float):
        return 8
    elif isinstance(a, str):
        return 0
    else:
        return 0


def bytes_convert(byte):
    if byte < 1024:
        return f"{byte} B"
    elif byte < 1024 ** 2:
        return f"{byte / 1024} KB"
    elif byte < 1024 ** 3:
        return f"{byte / 1024 ** 2} MB"
    elif byte < 1024 ** 4:
        return f"{byte / 1024 ** 3} GB"
    else:
        return f"{byte / 1024 ** 4} TB"


def comm_count(communicator, func, *args):
    now_comm_rounds = communicator.comm_rounds['send']
    now_comm_bytes = communicator.comm_bytes['send']
    func(*args)
    print(f"\033[33m Communication info of {func.__name__}: \033[0m")
    print(f"\033[33m Comm_rounds: {communicator.comm_rounds['send'] - now_comm_rounds}\033[0m")
    print(f"\033[33m Comm_costs: {bytes_convert(communicator.comm_bytes['send'] - now_comm_bytes)}\033[0m")
