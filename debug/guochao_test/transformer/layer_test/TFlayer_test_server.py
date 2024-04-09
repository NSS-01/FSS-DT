from application.Transformer.train.model.PretrainModel import PretrainModel
from application.Transformer.train.model.test_model import model
from application.NN.model.model_converter import load_secure_model, share_model
from application.NN.NNCS import NeuralNetworkCS
from config.base_configs import *


if __name__ == '__main__':
    # 初始化参数
    server = NeuralNetworkCS(type='server')
    server.inference_transformer(is_transformer=True)
    server.connect(('127.0.0.1', 8089), ('127.0.0.1', 8088), ('127.0.0.1', 20000), ('127.0.0.1', 20001))

    # sever端分享模型

    net = model

    share_model(net, "application/Transformer/train/model_save/model.pt",  model_file_path)

    # 保证两方通信
    net = load_secure_model(net, model_file_path, server)

    server.inference(net, 1)
    torch.cuda.empty_cache()
