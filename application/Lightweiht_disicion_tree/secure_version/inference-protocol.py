import time

import math
import numpy as np
import torch

from NssMPC.secure_model.mpc_party import SemiHonestCS

from NssMPC.crypto.primitives import ArithmeticSecretSharing
from NssMPC.common.ring.ring_tensor import RingTensor
from until import path, logger
class Tree:
    def __init__(self, m=-1, t=-1):
        self.m = m
        self.t = t

    def __str__(self):
        return f"f={self.m}, t={self.t}"
def inference(x, tree_list, tree_indices, p,c, h, party):
    if len(tree_list) == 0:
        return []
    if len(tree_list) == 1:
        if isinstance(tree_list.t,RingTensor):
            y_share = ArithmeticSecretSharing(tree_list.t, party=party)
            ground_y = y_share.restore().convert_to_real_field()
            logger.info(f"inference result:{ground_y}")
        return ground_y
    p = path(x, tree_list, tree_indices, p, h)
    p = RingTensor(torch.tensor(p,dtype=torch.int64))
    zore = RingTensor.zeros_like(p)

    label = ArithmeticSecretSharing(RingTensor.stack(c), party=party)
    y = label.restore().convert_to_real_field()
    if party.party_id==1:
        p0 =ArithmeticSecretSharing(p, party=party)
        p1 = ArithmeticSecretSharing(zore,party=party)
    else:
        p0 = ArithmeticSecretSharing(zore, party=party)
        p1 = ArithmeticSecretSharing(p, party=party)
    res = (p0*p1*label).sum()
    return res

def setup_server(server_model, X_test,y_test):
    server = SemiHonestCS(type='server')
    server.set_comparison_provider()
    server.set_multiplication_provider()
    server.online()

    n = len(server_model)
    h = int(math.log2(n+1))
    tree_indices = torch.arange(0,n)
    p = np.ones(2 ** (h - 1))
    c = [node.t for node in server_model[n//2:]]
    st = time.time()
    for i, x in enumerate(X_test):
        res = inference(x, server_model, tree_indices, p, c, h, server)
        res = res.restore().convert_to_real_field()
        logger.info(f"the {i}-th sample----- predicted label: {res}, ground-label:{y_test[i]},-----{ 'Yes' if y_test[i] ==res else 'No' }")
    et = time.time()
    logger.info(f"sever--inference running time:{(et - st) / len(X_test) * 1000:.1f} ms")
    server.close()


def setup_client(client_model,X_test,y_test):
    client = SemiHonestCS(type='client')
    client.set_comparison_provider()
    client.set_multiplication_provider()

    # p, tree_indices, c = init_inference(T, h)
    client.online()

    n = len(client_model)
    h = int(math.log2(n + 1))
    tree_indices = torch.arange(0, n)
    p = np.ones(2**(h-1))
    c = [node.t for node in client_model[n // 2:]]
    st = time.time()
    for i, x in enumerate(X_test):
        res =inference(x, client_model, tree_indices, p, c, h, client)
        res = res.restore().convert_to_real_field()
        logger.info(f"the {i}-th sample predicts label{res}, ground-label:{y_test[i]}")
    et = time.time()
    logger.info(f"client--inference running time:{(et-st)/len(X_test)*1000:.1f} ms")
    client.close()


if __name__ == '__main__':
    import threading

    data_name = 'bank_marketing'
    # data_name = 'skin'
    server_model = torch.load(f"../model/{data_name}_server_model.pth")
    # print(f"server--- tree{[(node.m, node.t) for node in server_model]}")
    client_model = torch.load(f"../model/{data_name}_client_model.pth")
    # print(f"client----tree{[(node.m, node.t) for node in client_model]}")


    _, X_test, _, y_test = torch.load(f'../data/{data_name}.pth')
    # Creating threads for secure_version and server
    client_thread = threading.Thread(target=setup_client,args=(client_model,torch.tensor(X_test),y_test), name="client")
    server_thread = threading.Thread(target=setup_server,args=(server_model,torch.tensor(X_test),y_test), name="server")
    # Starting threads
    client_thread.start()
    server_thread.start()

    # Optionally, wait for both threads to finish
    client_thread.join()
    server_thread.join()
