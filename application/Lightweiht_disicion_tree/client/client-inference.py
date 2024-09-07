import time

import math
import numpy as np

from utils import get_subtree_leaves

from config.base_configs import *
from crypto.mpc.semi_honest_party import SemiHonestCS
from crypto.tensor.ArithmeticSharedRingTensor import ArithmeticSharedRingTensor
from crypto.primitives.beaver.beaver_triples import BeaverBuffer
from crypto.tensor.RingTensor import RingTensor



"init "
client = SemiHonestCS(type='client')
client.set_beaver_provider(BeaverBuffer(client))
client.set_compare_key_provider()
client.beaver_provider.load_param()
client.connect(('127.0.0.1', 20000), ('127.0.0.1', 20001), ('127.0.0.1', 8089), ('127.0.0.1', 8088))
party = client



class Tree:
    def __init__(self, f=-1, t=-1):
        self.f = f
        self.t = t

    def __str__(self):
        return f"f={self.f}, t={self.t}"


def init_inference(T, h=1):
    l = 2 ** (h - 1) - 1
    r = 2 ** h - 1
    tree_indices = [x for x in range(0, 2 ** h - 1)]
    p = np.ones(2 ** (h - 1))
    c = np.array([0 for leaf in T[l:r]])

    return p, tree_indices,c


def path(x, tree_list, tree_indices, p, h):
    start_index = 0
    delta = 2 ** (h - 1) - 1
    queue = [start_index]
    while queue:
        current_index = queue.pop(0)
        left = 2 * current_index + 1
        right = 2 * current_index + 2
        if tree_list[current_index].f == -1:
            if left < 2 ** (h - 1) - 1:
                queue.append(left)
            if right < 2 ** (h - 1) - 1:
                queue.append(right)
        else:
            left_index, right_index = get_subtree_leaves(tree_indices, current_index)
            l = np.array(left_index) - delta
            r = np.array(right_index) - delta
            if x[tree_list[current_index].f]<tree_list[current_index].t:
                p[l.tolist()] = 1
                p[r.tolist()] = 0
                if left < 2 ** (h - 1) - 1:
                    queue.append(left)
            else:
                p[r.tolist()] = 1
                p[l.tolist()] = 0
                if right < 2 ** (h - 1) - 1:
                    queue.append(right)
    return p


def inference(x, tree_list, tree_indices, p,c, h):
    if len(tree_list) == 0:
        return []
    if len(tree_list) == 1:
        return tree_list[0].t
    p = path(x, tree_list, tree_indices, p, h)
    # print(p)
    zore = np.zeros_like(p)
    label = ArithmeticSharedRingTensor(torch.tensor(c,dtype=torch.int), party=party)
    p0 =ArithmeticSharedRingTensor(torch.tensor(p,dtype=torch.int), party=party)
    p1 = ArithmeticSharedRingTensor(torch.tensor(zore,dtype=torch.int),party=party)
    res = (p0*p1*label).sum()
    return res

# def inference_path()

if __name__ == "__main__":
    h = 4
    tree_indices = [x for x in range(0, 2 ** h - 1)]
    T = [Tree(f=x,t=1) for x in range(0, 2 ** h - 1)]
    h = int(math.log(len(T) + 1, 2))

    x = [x for x in range(0, 2 ** h - 1)]
    x[2]=-100
    # print(c[-1])
    T[0].f = -1
    T[5].f = -1
    st = time.time()
    p, tree_indices, c = init_inference(T, h)

    res = inference(x,T, tree_indices, p,c,h)
    et = time.time()
    # print(res)
    print(et-st)
    print(res.restore().convert_to_real_field())
    party.close()