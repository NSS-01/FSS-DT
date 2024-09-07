import time

from common.network.async_tcp import TCPServer
from common.utils.timer import get_time

DEVICE = 'cpu'
TEST_SERVER_ADDRESS = '127.0.0.1'
TEST_SERVER_PORT = 8989

tcp = TCPServer(TEST_SERVER_ADDRESS, TEST_SERVER_PORT)
tcp.run()
time.sleep(3)
print(get_time(tcp.receive_serializable_item_from, ('127.0.0.1', 1000)))
