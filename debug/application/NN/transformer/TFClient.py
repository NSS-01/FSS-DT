from torch.utils.data import DataLoader
from application.Transformer.bert.model.transformer import Transformer
from application.NN.model.model_converter import load_secure_model
from common.utils import get_time, comm_count
from config.base_configs import *
from application.NN.NNCS import NeuralNetworkCS
import torch.nn.functional as F
from application.Transformer.bert.train import CoupleDataset
from application.Transformer.bert.model.model_config import ModelConfig
from debug.crypto.mpc.party.configs import *

port_shift = 1
server_ip = '127.0.0.1'
server_server_address = (server_ip, server_server_port + port_shift)
server_client_address = (server_ip, server_client_port + port_shift)

client_server_address = (server_ip, client_server_port + port_shift)
client_client_address = (server_ip, client_client_port + port_shift)

if __name__ == '__main__':
    client = NeuralNetworkCS(type='client')
    client.inference_transformer(True)
    client.connect(client_server_address, client_client_address, server_server_address, server_client_address)

    # client端分享数据
    data = CoupleDataset(path="application\\Transformer\\bert\\data\\test")
    loader = DataLoader(data, batch_size=1)

    # config 配置文件
    config = ModelConfig(device=torch.device("cuda:0"))
    config.load("application\\Transformer\\bert\\model_save\\config.json")

    net = Transformer

    # 保证两方通信
    net = load_secure_model(net, model_file_path, client)

    for src_ids, trg_ids in loader:
        src_ids = src_ids.to(config.device)
        trg_ids = trg_ids.to(config.device)
        src_position_ids = torch.arange(src_ids.size(1), dtype=torch.long, device=config.device)
        trg_position_ids = torch.arange(trg_ids[:, :-1].size(1), dtype=torch.long, device=config.device)
        src_zero_ids = torch.zeros_like(src_ids, device=config.device)
        trg_zero_ids = torch.zeros_like(trg_ids[:, :-1], device=config.device)

        src_ids = F.one_hot(src_ids, config.vocab_size) * 1.0
        trg_ids = F.one_hot(trg_ids[:, :-1], config.vocab_size) * 1.0
        src_position_ids = F.one_hot(src_position_ids, config.max_len) * 1.0
        trg_position_ids = F.one_hot(trg_position_ids, config.max_len) * 1.0
        src_zero_ids = F.one_hot(src_zero_ids, 2) * 1.0
        trg_zero_ids = F.one_hot(trg_zero_ids, 2) * 1.0

        print(src_ids.shape)
        p = comm_count(client, get_time, client.inference, net, src_ids, src_position_ids, src_zero_ids, trg_ids,
                       trg_position_ids, trg_zero_ids)
        pids = torch.argmax(p, dim=2)

        y = trg_ids[:, 1:]
        ss = data.tokenizer.decode(src_ids.detach().cpu().numpy()[0])
        p_s = data.tokenizer.decode(pids.detach().cpu().numpy()[0])
        ori_s = data.tokenizer.decode(y.detach().cpu().numpy()[0])
        print("\n\n==[{}]==\n==[{}]==\n==[{}]==\n\n".format(ss, p_s, ori_s))

    client.close()
    torch.cuda.empty_cache()
