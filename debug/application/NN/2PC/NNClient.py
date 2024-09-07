import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset

from application.NN.AlexNet.Alexnet import AlexNet
from application.NN.ResNet.ResNet import resnet50
from application.NN.model.model_converter import load_secure_model
from common.utils.timer import get_time
from config.base_configs import *
from application.NN.NNCS import NeuralNetworkCS

from debug.crypto.mpc.party.configs import *

port_shift = 12
server_ip = '127.0.0.1'
server_server_address = (server_ip, server_server_port + port_shift)
server_client_address = (server_ip, server_client_port + port_shift)

client_server_address = (server_ip, client_server_port + port_shift)
client_client_address = (server_ip, client_client_port + port_shift)


if __name__ == '__main__':

    client = NeuralNetworkCS(type='client')
    client.connect(client_server_address, client_client_address, server_server_address, server_client_address)

    # client端分享数据
    transform1 = transforms.Compose([transforms.ToTensor()])
    test_set = torchvision.datasets.MNIST(root='data/NN/', train=False, download=True, transform=transform1)

    indices = list(range(10))
    subset_data = Subset(test_set, indices)
    test_loader = torch.utils.data.DataLoader(subset_data, batch_size=1, shuffle=False, num_workers=0)

    net = AlexNet
    # net = resnet50

    num = client.dummy_model(test_loader)
    # 保证两方通信
    # client.receive_tensor()
    net = load_secure_model(net, model_file_path, client)
    # client.receive_tensor()

    correct_total = 0
    total_total = 0

    for data in test_loader:
        correct = 0
        total = 0
        images, labels = data

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        res = get_time(client.inference, net, images)

        _, predicted = torch.max(res, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        total_total += total
        correct_total += correct

        print('Accuracy of the network on test images:{}%'.format(100 * correct / total))  # 输出识别准确率
        # print("总共用时：", end_time - start_time)

    print('Accuracy of the network on test images:{}%'.format(100 * correct_total / total_total))  # 输出识别准确率

    client.close()
