import time

import torch

from common.network.mt_tcp import TCPServer
from common.utils.timer import get_time

DEVICE = 'cpu'
TEST_SERVER_ADDRESS = '127.0.0.1'
TEST_SERVER_PORT = 8089

tcp = TCPServer(TEST_SERVER_ADDRESS, TEST_SERVER_PORT)
tcp.start()

tcp.run()

time.sleep(3)
print(get_time(tcp.recv_from, ('127.0.0.1', 11000)))

tcp.listening_thread.join()
