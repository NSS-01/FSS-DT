import time

import math
import numpy as np
import torch

from NssMPC.secure_model.mpc_party import SemiHonestCS

from NssMPC.crypto.primitives import ArithmeticSecretSharing
from NssMPC.common.ring.ring_tensor import RingTensor
from until import path,Tree


def init_inference(T, h=1):
    l = 2 ** (h - 1) - 1
    r = 2 ** h - 1
    tree_indices = [x for x in range(0, 2 ** h - 1)]
    p = np.ones(2 ** (h - 1))
    c = np.array([leaf.t for leaf in T[l:r]])

    return p, tree_indices,c
def inference(x, tree_list, tree_indices, p,c, h, party):
    if len(tree_list) == 0:
        return []
    if len(tree_list) == 1:
        return tree_list[0].t
    p = path(x, tree_list, tree_indices, p, h)
    # print(p)
    zore = np.zeros_like(p)
    label = ArithmeticSecretSharing(RingTensor.convert_to_ring(torch.tensor(c,dtype=torch.int)), party=party)
    p0 =ArithmeticSecretSharing(RingTensor.convert_to_ring(torch.tensor(p,dtype=torch.int)), party=party)
    p1 = ArithmeticSecretSharing(RingTensor.convert_to_ring(torch.tensor(zore,dtype=torch.int)),party=party)
    res = (p0*p1*label).sum()
    return res

def setup_server():
    server = SemiHonestCS(type='server')
    server.set_comparison_provider()
    server.set_multiplication_provider()
    server.online()
    h = 4
    tree_indices = [x for x in range(0, 2 ** h - 1)]
    T = [Tree(f=x, t=1) for x in range(0, 2 ** h - 1)]
    h = int(math.log(len(T) + 1, 2))

    x = [x for x in range(0, 2 ** h - 1)]
    x[2] = -100
    # print(c[-1])
    T[0].f = -1
    T[5].f = -1
    p, tree_indices, c = init_inference(T, h)
    inference(x, T, tree_indices, p,c, h, server)
    server.close()


def setup_client():
    client = SemiHonestCS(type='client')
    client.set_comparison_provider()
    client.set_multiplication_provider()
    h = 4
    tree_indices = [x for x in range(0, 2 ** h - 1)]
    T = [Tree(f=x, t=1) for x in range(0, 2 ** h - 1)]
    h = int(math.log(len(T) + 1, 2))

    x = [x for x in range(0, 2 ** h - 1)]
    x[2] = -100
    # print(c[-1])
    T[0].f = -1
    T[5].f = -1
    p, tree_indices, c = init_inference(T, h)
    client.online()


    st = time.time()
    inference(x, T, tree_indices, p, c, h, client)
    et = time.time()
    print(f"runing time: {et-st}")
    client.close()


if __name__ == '__main__':
    import threading
    # Creating threads for secure_version and server
    client_thread = threading.Thread(target=setup_client)
    server_thread = threading.Thread(target=setup_server)
    # Starting threads
    client_thread.start()
    server_thread.start()

    # Optionally, wait for both threads to finish
    client_thread.join()
    server_thread.join()
