import time

import torch

from torch_dist import Agent

world_size = 2  # 总共有两个进程/计算机
agent = Agent(1, world_size)
print(1)
tensor = torch.empty([1000000, 2], dtype=torch.int64)
start = time.time()
# while 1:
agent.recv_tensor(tensor, src=0)
print(f"Process 1 received tensor: {tensor}")
agent.cleanup()
end = time.time()
print(end - start)
