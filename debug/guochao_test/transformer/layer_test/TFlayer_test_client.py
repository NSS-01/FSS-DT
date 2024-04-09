from application.Transformer.train.model.test_model import model
from application.NN.model.model_converter import load_secure_model
from config.base_configs import *
from application.NN.NNCS import NeuralNetworkCS

if __name__ == '__main__':
    client = NeuralNetworkCS(type='client')
    client.inference_transformer(is_transformer=True)
    client.connect(('127.0.0.1', 20000), ('127.0.0.1', 20001), ('127.0.0.1', 8089), ('127.0.0.1', 8088))

    # client端分享数据

    inputs = torch.tensor([[-2., 2., 7.], [-9., 9., 0.]], device=DEVICE)

    net = model

    # 保证两方通信
    net = load_secure_model(net, model_file_path, client)
    res = client.inference(net, inputs)

    print(res)

    client.close()
