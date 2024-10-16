import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
from application.neural_network.party import HonestMajorityNeuralNetWork3PC
import torch
import NssMPC.application.neural_network as nn
from NssMPC.common.utils import comm_count
from data.AlexNet.Alexnet import AlexNet
from NssMPC.config import DEVICE, NN_path
from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional import *
# 测试恶意乘法
secure_model = 0

if secure_model == 0:
    from application.neural_network.party import HonestMajorityNeuralNetWork3PC as Party
else:
    from application.neural_network.party import SemiHonest3PCParty as Party

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
#         self.conv1 = torch.nn.ReLU()
#
#     def forward(self, x):
#         x = self.conv1(x)
#         return x


# 测试恶意乘法
if __name__ == '__main__':
    P = Party(1)
    P.set_comparison_provider()
    P.set_multiplication_provider()
    P.set_trunc_provider()
    P.online()
    net = AlexNet()
    # net = Net()
    print("接收权重")
    local_param = P.receive(0)


    print("预处理一些东西")
    num = P.dummy_model()
    net = nn.utils.load_model(net, local_param)
    print("接收输入")

    share_input = receive_share_from(0, P)
    # print("share input", share_input.restore().convert_to_real_field())
    for i in range(3):
        output = comm_count(P.communicator, net, share_input)
    # print("output", output.restore().convert_to_real_field())
