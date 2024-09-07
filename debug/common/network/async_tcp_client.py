import torch

from common.network.async_tcp import TCPClient

DEVICE = 'cpu'
TEST_CLIENT_ADDRESS = '127.0.0.1'
TEST_CLIENT_PORT = 1000

tcp = TCPClient(TEST_CLIENT_ADDRESS, TEST_CLIENT_PORT)
tcp.connect_to('127.0.0.1', 8989)

tcp.send_serializable_item(torch.ones(1024 * 1024 * 1024))
