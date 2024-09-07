import pickle

import torch

from common.network.mt_tcp import TCPClient

DEVICE = 'cpu'
TEST_CLIENT_ADDRESS = '127.0.0.1'
TEST_CLIENT_PORT = 11000

tcp = TCPClient(TEST_CLIENT_ADDRESS, TEST_CLIENT_PORT)
tcp.start()

tcp.connect_to('127.0.0.1', 8089)
tcp.send_msg(pickle.dumps(torch.ones(1024 * 1024 * 1024)))
