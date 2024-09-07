from application.Transformer.bert.model.transformer import Transformer

from application.NN.model.model_converter import load_secure_model, share_model
from application.NN.NNCS import NeuralNetworkCS
from config.base_configs import *
from debug.crypto.mpc.party.configs import *

port_shift = 1
server_ip = '127.0.0.1'
server_server_address = (server_ip, server_server_port + port_shift)
server_client_address = (server_ip, server_client_port + port_shift)

client_server_address = (server_ip, client_server_port + port_shift)
client_client_address = (server_ip, client_client_port + port_shift)

if __name__ == '__main__':
    # 初始化参数
    server = NeuralNetworkCS(type='server')
    server.inference_transformer(True)
    server.connect(server_server_address, server_client_address, client_server_address, client_client_address)

    # sever端分享模型
    net = Transformer

    share_model(net, "application/Transformer/bert/model_save/c_model_done.pt", model_file_path)

    # 保证两方通信
    net = load_secure_model(net, model_file_path, server)

    server.inference(net, 6)
    torch.cuda.empty_cache()
