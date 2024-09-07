import time
from typing import Tuple

import torch

from application.Lightweiht_disicion_tree.config_tree import TreeNode
from application.Lightweiht_disicion_tree.config_tree import m, m0, server_data, unique_values_by_column

from config.base_configs import *
from crypto.mpc.semi_honest_party import SemiHonestCS
from crypto.tensor.ArithmeticSharedRingTensor import ArithmeticSharedRingTensor
from crypto.primitives.beaver.beaver_triples import BeaverBuffer
from crypto.tensor.RingTensor import RingTensor


def _max_with_index(x,indices,dim=0):
    def max_(inputs, indices):
        # 初始化索引
        if inputs.shape[0] == 1:
            return inputs, indices
        if inputs.shape[0] % 2 == 1:
            inputs_ = inputs[-1:]
            inputs = ArithmeticSharedRingTensor.cat([inputs, inputs_], 0)
            indices_ = indices[-1:]
            indices = ArithmeticSharedRingTensor.cat([indices, indices_], 0)
        inputs_0 = inputs[0::2]
        inputs_1 = inputs[1::2]
        indices_0 = indices[0::2]
        indices_1 = indices[1::2]

        ge = inputs_0 >= inputs_1
        le = (ge - RingTensor.ones_like(ge)) * -1
        max_values = ge * inputs_0 + le * inputs_1
        max_indices = ge * indices_0 + le * indices_1
        return max_values, max_indices  # 仅保留胜出的索引

    if dim is None:
        x = x.flatten()
        indices = indices.flatten()
    else:
        x = x.transpose(dim, 0)
        indices = indices.transpose(dim, 0)
    if x.shape[0] == 1:
        return x.transpose(dim, 0).squeeze(-1), indices.transpose(dim, 0).squeeze(-1)
    else:
        x, indices = max_(x, indices)
    # 需要适当调整最后的转置和索引选择以匹配原始维度
    return _max_with_index(x.transpose(0, dim), indices.transpose(0, dim),dim)



def _max_with_index_threshold(x,indices,thresholds,dim=0):
    def max_(inputs,indices,thresholds):
        # 初始化索引
        if inputs.shape[0] == 1:
            return inputs, indices,thresholds
        if inputs.shape[0] % 2 == 1:
            inputs_ = inputs[-1:]
            inputs = ArithmeticSharedRingTensor.cat([inputs, inputs_], 0)
            indices_ = indices[-1:]
            indices = ArithmeticSharedRingTensor.cat([indices, indices_], 0)
            thresholds_ = thresholds[-1:]
            thresholds = ArithmeticSharedRingTensor.cat([thresholds,thresholds_],0)


        inputs_0 = inputs[0::2]
        inputs_1 = inputs[1::2]
        indices_0 = indices[0::2]
        indices_1 = indices[1::2]
        thresholds_0 = thresholds[0::2]
        thresholds_1 = thresholds[1::2]
        ge = inputs_0 >= inputs_1
        le = (ge - RingTensor.ones_like(ge)) * -1
        max_values = ge * inputs_0 + le * inputs_1
        max_indices = ge * indices_0 + le * indices_1
        max_thresholds = ge*thresholds_0+le*thresholds_1
        return max_values, max_indices,max_thresholds  # 仅保留胜出的索引

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
        x, indices,thresholds = max_(x, indices,thresholds)
    # 需要适当调整最后的转置和索引选择以匹配原始维度
    return _max_with_index_threshold(x.transpose(0, dim),indices.transpose(0, dim),thresholds.transpose(0,dim),dim)


data = server_data
Thresholds = unique_values_by_column(data)
labels = Thresholds[-1].view(-1)

num_labels = len(labels)
n_samples = data.size(0)
data_labels = data[:, -1].unsqueeze(1)
unique_labels = labels.squeeze().unsqueeze(0)
binary_matrix = (data_labels == unique_labels)
server = SemiHonestCS(type='server')
server.set_beaver_provider(BeaverBuffer(server))
server.set_compare_key_provider()
server.beaver_provider.load_param()
server.connect(('127.0.0.1', 8089), ('127.0.0.1',8888), ('127.0.0.1', 20000), ('127.0.0.1', 20001))
data = server_data
party=server
zero = ArithmeticSharedRingTensor(torch.tensor(0), party=party)
g = ArithmeticSharedRingTensor(torch.tensor(0, dtype=torch.int64), party=party)
nk = ArithmeticSharedRingTensor(torch.tensor(0, dtype=torch.int64, device=DEVICE), party=party)
ny = ArithmeticSharedRingTensor(binary_matrix*party.party_id, dtype="int", party=party)

def SecureBuildTree(data, share_w: ArithmeticSharedRingTensor, T: list, h: int,Maximum_depth:int):
    if h in range(0, 2 ** (Maximum_depth-1) - 1):
        share_m, share_t = SecureFindBestSplit(data, share_w)
        c = ((share_m - RingTensor(m0,dtype=int,device=DEVICE)) >= zero).restore().convert_to_real_field()
        m1 = share_m.restore().convert_to_real_field()
        z = (share_w.party.party_id == c)
        T[h].m = z * m1.int() - (1 - z.int())
        t = share_t.restore().convert_to_real_field()
        T[h].t = z * t - (1 - z.int())
        # print(data[:, m1.int()]-t)
        b_ready = (data[:, m1.int()]-t>0) * share_w.party.party_id
        b = ArithmeticSharedRingTensor(b_ready, dtype=torch.int, party=share_w.party)
        w2h = share_w * b
        w2h1 = share_w*(-b+RingTensor(1, dtype=int, device=DEVICE))
        SecureBuildTree(data, w2h, T, 2 * h + 1,Maximum_depth)
        SecureBuildTree(data, w2h1, T, 2 * h + 2,Maximum_depth)
    elif h in range(2 **(Maximum_depth-1) - 1, 2 ** Maximum_depth - 1):
        y_sum = (share_w.view(n_samples, -1) * ny).sum(dim=0)

        share_labels = ArithmeticSharedRingTensor(labels * share_w.party.party_id, party=share_w.party)
        _, share_y = _max_with_index(y_sum, share_labels)
        T[h].t = share_y
        T[h].m = "leaf node"


def SecureFindBestSplit(data, share_w: ArithmeticSharedRingTensor) -> Tuple[
    ArithmeticSharedRingTensor, ArithmeticSharedRingTensor]:
    each_index_ts = []
    each_best_gs = []
    for j in range(0, m - 1):

        feature_column = data[:, j].unsqueeze(1)
        thresholds = Thresholds[j].view(1,-1)


        comparison_matrix = feature_column > thresholds
        # client 0, server: 1
        if j > m0 - 1:
            b = ArithmeticSharedRingTensor(comparison_matrix * share_w.party.party_id,  party=share_w.party)
        else:
            b = ArithmeticSharedRingTensor(comparison_matrix * (1 - share_w.party.party_id),
                                           party=share_w.party)
        share_wl_matrix = share_w.view(n_samples, -1) * b
        share_wr_matrix = share_w.view(n_samples,-1) *(b-RingTensor(1))
        Dls = share_wl_matrix.sum(dim=0)
        Drs = share_wr_matrix.sum(dim=0)
        Dlk = ny.view(-1, n_samples) @ share_wl_matrix
        Dlk2 = (Dlk * Dlk).sum(dim=0)
        Drk = ny.view(-1, n_samples) @ share_wr_matrix
        Drk2 = (Drk * Drk).sum(dim=0)

        gs = Drs * (Dls - Dlk2) + Dls * (Dls - Drk2)
        del Dlk, Dlk2, Drk, Drk2
        t = thresholds.view(-1)
        ts = ArithmeticSharedRingTensor(t* share_w.party.party_id,party=share_w.party)
        g, t = _max_with_index(gs, ts)
        each_index_ts.append(t.view(-1))
        each_best_gs.append(g.view(-1))
    ms = ArithmeticSharedRingTensor(torch.arange(m - 1) * share_w.party.party_id, party=share_w.party)
    best_gs = ArithmeticSharedRingTensor.cat(each_best_gs, 0)
    best_ts = ArithmeticSharedRingTensor.cat(each_index_ts, 0)
    print(best_ts.shape, ms.shape, best_gs.shape)
    _, m_, t = _max_with_index_threshold(best_gs, ms, best_ts)
    return m_, t
if __name__ == "__main__":
    # log = []
    Maximum_depth = 4
    T = [TreeNode() for _ in range(0, 2 ** Maximum_depth - 1)]
    w1 = ArithmeticSharedRingTensor(torch.ones(n_samples, dtype=torch.int), party=party)
    st = time.time()
    SecureBuildTree(server_data, share_w=w1, T=T, h=0, Maximum_depth=Maximum_depth)
    et = time.time()
    print(f" running time:{et - st}")
    # log.append(f"running time:{et - st}")
    # file_path = 'application/Lightweiht_disicion_tree/log/server_log.txt'
    # with open(file_path, 'w') as file:
    #     for item in log:
    #         file.write("%s\n" % item)

    party.close()