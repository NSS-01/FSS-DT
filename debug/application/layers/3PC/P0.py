import time

import torch.utils.data

from NssMPC import RingTensor
import torch
import NssMPC.application.neural_network as nn
from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional import share
from data.AlexNet.Alexnet import AlexNet


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        return x


# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = torch.nn.Conv2d(1, 20, kernel_size=5, padding=1)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         return x



secure_model = 0
if secure_model == 0:
    from application.neural_network.party import HonestMajorityNeuralNetWork3PC as Party
else:
    from application.neural_network.party import SemiHonest3PCParty as Party


if __name__ == '__main__':
    P = Party(0)
    P.set_comparison_provider()
    P.set_multiplication_provider()
    P.set_trunc_provider()
    P.online()

    test_input = torch.randint(-10, 10, [2,3,33,33]) * 1.0
    # test_input = torch.randint(-10, 10, [400, 400]) * 1.0

    # print("test_input:", test_input)
    net = AlexNet()
    # net = Net()
    # test_output = net(test_input)
    # print("test_output", test_output)

    print("开始分享权重")
    shared_param = nn.utils.share_model(net, share_type=32)
    local_param = shared_param[0]
    P1_param = shared_param[1]
    P2_param = shared_param[2]
    P.send(1, P1_param)
    P.send(2, P2_param)

    print("预处理一些东西")
    num = P.dummy_model(net, test_input)
    net = nn.utils.load_model(net, local_param)


    print("开始分享输入")

    share_input = share(RingTensor.convert_to_ring(test_input), P)
    # print("share input", share_input.restore().convert_to_real_field())

    for i in range(3):
        st = time.time()
        output = net(share_input)
        et = time.time()
        torch.cuda.empty_cache()
        print("time cost", et-st)
    # print("output", output.restore().convert_to_real_field())