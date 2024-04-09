import time

import torch

from torch_dist import Agent

world_size = 2  # 总共有两个进程/计算机
agent = Agent(0, world_size)
import torch.distributed as dist
print(dist.is_initialized())
tensor = torch.randint(-10, 0, [1000000, 2], dtype=torch.int64)
start = time.time()
# while(1):
agent.send_tensor(tensor, dst=1)
# agent.recv_tensor(tensor, src=1)
agent.cleanup()
end = time.time()
print(end - start)
