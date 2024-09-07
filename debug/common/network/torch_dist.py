import torch.distributed as dist


class Agent(object):
    def __init__(self, rank, world_size, ip='192.168.15.2', port='20000'):
        '''
        :param rank: 参与方id
        :param world_size: 参与方数量
        :param ip: 主机ip
        :param port: 主机端口
        '''
        self.rank = rank
        dist.init_process_group(
            backend='gloo',
            init_method=f'tcp://{ip}:{port}',
            rank=self.rank,
            world_size=world_size
        )

    @staticmethod
    def send_tensor(tensor, dst):
        dist.send(tensor=tensor, dst=dst)

    @staticmethod
    def recv_tensor(tensor, src):
        dist.recv(tensor=tensor, src=src)

    def cleanup(self):
        dist.destroy_process_group()
