import time

import torch
from NssMPC.secure_model.mpc_party import SemiHonestCS
from NssMPC.crypto.primitives import ArithmeticSecretSharing
from NssMPC.common.ring.ring_tensor import RingTensor
import warnings
warnings.filterwarnings("ignore")
def _min_with_index(x,indices,dim=0):
    def min_(inputs, indices):
        # 初始化索引
        if inputs.shape[0] == 1:
            return inputs, indices
        if inputs.shape[0] % 2 == 1:
            inputs_ = inputs[-1:]
            inputs = ArithmeticSecretSharing.cat([inputs, inputs_], 0)
            indices_ = indices[-1:]
            indices = ArithmeticSecretSharing.cat([indices, indices_], 0)
        inputs_0 = inputs[0::2]
        inputs_1 = inputs[1::2]
        indices_0 = indices[0::2]
        indices_1 = indices[1::2]

        b = inputs_0<= inputs_1 # x<y=b
        min_values = b * inputs_0 + (1-b)* inputs_1
        min_indices = b * indices_0 + (1-b) * indices_1
        return min_values, min_indices  # 仅保留胜出的索引

    if dim is None:
        x = x.flatten()
        indices = indices.flatten()
    else:
        x = x.transpose(dim, 0)
        indices = indices.transpose(dim, 0)
    if x.shape[0] == 1:
        return x.transpose(dim, 0).squeeze(-1), indices.transpose(dim, 0).squeeze(-1)
    else:
        x, indices = min_(x, indices)
    # 需要适当调整最后的转置和索引选择以匹配原始维度
    return _min_with_index(x.transpose(0, dim), indices.transpose(0, dim),dim)

def _min_with_index_threshold(x,indices,thresholds,dim=0):
    def min_(inputs,indices,thresholds):
        # 初始化索引
        if inputs.shape[0] == 1:
            return inputs, indices,thresholds
        if inputs.shape[0] % 2 == 1:
            inputs_ = inputs[-1:]
            inputs = ArithmeticSecretSharing.cat([inputs, inputs_], 0)
            indices_ = indices[-1:]
            indices = ArithmeticSecretSharing.cat([indices, indices_], 0)
            thresholds_ = thresholds[-1:]
            thresholds = ArithmeticSecretSharing.cat([thresholds,thresholds_],0)


        inputs_0 = inputs[0::2]
        inputs_1 = inputs[1::2]
        indices_0 = indices[0::2]
        indices_1 = indices[1::2]
        thresholds_0 = thresholds[0::2]
        thresholds_1 = thresholds[1::2]
        b = inputs_0< inputs_1 # x<y=b
        # bs =b.restore().convert_to_real_field()
        min_values = b * inputs_0 + (1-b) * inputs_1
        min_indices = b * indices_0 + (1-b) * indices_1
        min_thresholds = b*thresholds_0+(1-b)*thresholds_1
        return min_values, min_indices,min_thresholds  # 仅保留胜出的索引

    if dim is None:
        x = x.flatten()
        indices = indices.flatten()
        thresholds = thresholds.flatten()
    else:
        x = x.transpose(dim, 0)
        indices = indices.transpose(dim, 0)
        thresholds = thresholds.transpose(dim, 0)
    if x.shape[0] == 1:
        return x.transpose(dim, 0).squeeze(-1), indices.transpose(dim, 0).squeeze(-1), thresholds.transpose(dim, 0).squeeze(-1)
    else:
        x, indices,thresholds = min_(x, indices,thresholds)
    # 需要适当调整最后的转置和索引选择以匹配原始维度
    return _min_with_index_threshold(x.transpose(0, dim),indices.transpose(0, dim),thresholds.transpose(0,dim),dim)

class Tree:
    def __init__(self, t=-1, m=-1):
        self.m = m
        self.t = t

    def __str__(self):
        return f"TreeList Object: m={self.m}, t={self.t}"

def SecureBuildTree(m0,data, thresholds, share_w: ArithmeticSecretSharing, T: list, h: int, maxinum_depth: int, ny:ArithmeticSecretSharing):
    if h in range(0, 2 ** (maxinum_depth - 1) - 1):
        share_m, share_t = SeureFindBestSplit(m0,data, share_w, thresholds,ny)
        c = (share_m-m0)>0
        c= c.restore().convert_to_real_field()
        m = share_m.restore().convert_to_real_field()
        t = share_t.restore().convert_to_real_field()
        z = (share_w.party.party_id == c)
        T[h].m = z * m.int() - (1 - z.int())
        T[h].t = z * t - (1 - z.int())
        if share_w.party.party_id==1:
            print(f"T[{h}].m ={T[h].m}\n T[{h}].t={T[h].t}")

        b_test = (data[:, m.int()]<=t) * share_w.party.party_id
        b = ArithmeticSecretSharing(RingTensor.convert_to_ring(b_test), party=share_w.party)
        wr = share_w * b
        wl = share_w*(1-b)
        SecureBuildTree(m0,data, thresholds,wl, T, 2*h+1, maxinum_depth=max_height,ny=ny)
        SecureBuildTree(m0, data, thresholds, wr, T,2*h+2, maxinum_depth=max_height, ny=ny)
    elif h in range(2 ** (maxinum_depth - 1) - 1, 2 ** maxinum_depth - 1):
        y_sum = share_w@ny
        y_sum = y_sum.squeeze(0)
        labels = thresholds[-1].view(-1)
        share_labels = ArithmeticSecretSharing(RingTensor(labels.to(torch.int64)* share_w.party.party_id), party=share_w.party)
        _, share_y = _min_with_index(-y_sum, share_labels)
        T[h].t = -share_y
        T[h].m = -2
def SeureFindBestSplit(m0,data, share_w, thresholds,ny):
    n, m = data.shape
    each_best_ts = []
    each_best_gs = []

    for j in range(0, m-1):
        threshold = thresholds[j].view(1, -1)
        feature_vector = data[:, j].unsqueeze(1)
        local_test = feature_vector<=threshold
        # server_id: 1 and client_id: 0
        if j < m0:
            share_local_test = ArithmeticSecretSharing(RingTensor(local_test * (1-share_w.party.party_id)),
                                                       party=share_w.party)
        else:
            share_local_test = ArithmeticSecretSharing(RingTensor((local_test * share_w.party.party_id)),party=share_w.party)
        wl_share = share_w.view(n,-1) * share_local_test
        wr_share = share_w.view(n,-1)*(1-share_local_test)
        num_Dl = wl_share.sum(dim=0)
        num_Dr = wr_share.sum(dim=0)


        Dlk = (ny.T)@wl_share
        Drk = (ny.T)@wr_share
        square_Dlk = (Dlk*Dlk).sum(dim=0)
        square_Drk = (Drk*Drk).sum(dim=0)

        t = threshold.to(torch.int).view(-1)
        ts = ArithmeticSecretSharing(RingTensor.convert_to_ring(t*share_w.party.party_id),party=share_w.party)
        gs = num_Dr * (num_Dl - square_Dlk) + num_Dl * (num_Dr - square_Drk)
        #gs = (num_Dl / (num_Dr+num_Dr)) * (square_Dlk / num_Dl)+(num_Dr/(num_Dr+num_Dr))*(square_Drk/num_Dr)
        g, t = _min_with_index(gs, ts)
        each_best_gs.append(g.view(-1))
        each_best_ts.append(t.view(-1))
    ms = ArithmeticSecretSharing(RingTensor(torch.arange(m - 1) * share_w.party.party_id),
                                party=share_w.party)
    best_gs = ArithmeticSecretSharing.cat(each_best_gs, dim=0)
    best_ts = ArithmeticSecretSharing.cat(each_best_ts, dim=0)
    _, best_m, best_t = _min_with_index_threshold(best_gs, ms, best_ts)

    return best_m,best_t


def setup_client(data_name, max_height):
    file_name = f'../data/{data_name}_client_data.pth'
    m0, thresholds, client_data = torch.load(file_name)
    n, m = client_data.shape
    num_class = len(thresholds[-1])
    client_tree_list = [Tree() for _ in range(2 ** max_height)]

    client = SemiHonestCS(type='client')
    client.set_comparison_provider()
    client.set_multiplication_provider()
    client.set_nonlinear_operation_provider()
    ny = ArithmeticSecretSharing(RingTensor.zeros(size=(n,num_class)), party=client)
    client.online()
    '''
    client training code
    '''
    root_share_w = ArithmeticSecretSharing(RingTensor.zeros(size=(n,)), party=client)
    SecureBuildTree(m0,client_data, thresholds, root_share_w, client_tree_list, h=0, maxinum_depth=max_height,ny=ny)
    client.close()

def setup_server(data_name, max_height):

    file_name = f'../data/{data_name}_server_data.pth'
    m0, thresholds, server_data = torch.load(file_name)
    server_tree_list = [Tree() for _ in range(2 ** max_height-1)]
    n, m = server_data.shape
    labels = thresholds[-1].view(1, -1)
    data_label = server_data[:,-1].unsqueeze(1)
    unique_label = labels.squeeze().unsqueeze(0)
    binary_matrix = data_label == unique_label
    server = SemiHonestCS(type='server')
    server.set_comparison_provider()
    server.set_nonlinear_operation_provider()
    server.set_multiplication_provider()
    ny = ArithmeticSecretSharing(RingTensor(binary_matrix+0),party=server)
    server.online()
    '''
    server training code
    '''
    root_share_w = ArithmeticSecretSharing(RingTensor(torch.ones(n,dtype=torch.int64)), party=server)
    SecureBuildTree(m0,server_data, thresholds, root_share_w, server_tree_list, h=0, maxinum_depth=max_height,ny=ny)
    server.close()
if __name__ == '__main__':
    import threading
    import  json
    import os
    import NssMPC.config.configs as cfg
    cfg.GE_TYPE = "SIGMA"
    data_name = 'mnist'
    max_height = 4
    st = time.time()
    # Creating threads for secure_version and server
    client_thread = threading.Thread(target=setup_client, args=(data_name, max_height),name="setup_client")
    server_thread = threading.Thread(target=setup_server, args=(data_name, max_height),name="setup_server")
    # Starting threads
    client_thread.start()
    server_thread.start()
    # Optionally, wait for both threads to finish
    client_thread.join()
    server_thread.join()
    et = time.time()
    print(f"runing_time:{et - st:.2f}")
