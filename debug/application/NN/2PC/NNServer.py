from application.NN.AlexNet.Alexnet import AlexNet
from application.NN.NNCS import NeuralNetworkCS
from application.NN.ResNet.ResNet import resnet50
from application.NN.model.model_converter import load_secure_model, share_model
from config.base_configs import *
from debug.crypto.mpc.party.configs import *

port_shift = 12
server_ip = '127.0.0.1'
server_server_address = (server_ip, server_server_port + port_shift)
server_client_address = (server_ip, server_client_port + port_shift)

client_server_address = (server_ip, client_server_port + port_shift)
client_client_address = (server_ip, client_client_port + port_shift)

if __name__ == '__main__':
    # 初始化参数
    server = NeuralNetworkCS(type='server')
    server.connect(server_server_address, server_client_address, client_server_address, client_client_address)

    # sever端分享模型
    net = AlexNet
    # net = resnet50

    # share_model(net, "application/NN/AlexNet/MNIST_bak.pkl", model_file_path)
    # share_model(net, "application/NN/ResNet/ResNet50.pkl", model_file_path)
    num = server.dummy_model(net)
    # num=10
    # 保证两方通信
    # server.send_tensor(torch.tensor(1))
    net = load_secure_model(net, model_file_path, server)
    # server.send_tensor(torch.tensor(1))

    while num:
        server.inference(net)
        torch.cuda.empty_cache()
        num -= 1
    server.close()
