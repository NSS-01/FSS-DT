import torch
DEVICE = 'cpu'
TEST_SERVER_ADDRESS = '127.0.0.1'
TEST_SERVER_PORT = 8989
from common.network.tcp import TCPClient

tcp = TCPClient(TEST_SERVER_ADDRESS, TEST_SERVER_PORT)

tcp.run()

test_client = torch.randint(0,10,[10],device=DEVICE)
print("client", test_client)

tcp.send_torch_array(test_client)
test_server = tcp.receive_torch_array()

print("server", test_server)