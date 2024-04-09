import torch
DEVICE = 'cpu'
TEST_SERVER_ADDRESS = '127.0.0.1'
TEST_SERVER_PORT = 8989
from common.network.tcp import TCPServer

tcp = TCPServer(TEST_SERVER_ADDRESS, TEST_SERVER_PORT)

tcp.run()

test_server = torch.randint(0,10,[10],device=DEVICE)
print("server", test_server)

tcp.send_torch_array(test_server)
test_client = tcp.receive_torch_array()

print("client", test_client)