import time

import torch.cuda

from NssMPC import RingTensor
# Falcon use uint32_t as base type

class Pair:
    def __init__(self):
        self.first = None
        self.second = None

    def make_pair(self, first, second):
        self.first = first
        self.second = second

    def random_pair(self, shape):
        self.first = RingTensor.random(shape)
        self.second = RingTensor.random(shape)

    def __getitem__(self, item):
        first = self.first[item]
        second = self.second[item]
        new_pair = Pair()
        new_pair.make_pair(first, second)
        return new_pair

    def __setitem__(self, key, value):
        self.first[key] = value.first
        self.second[key] = value.second

    def __add__(self, other):
        new_pair = Pair()
        new_pair.make_pair(self.first + other.first, self.second + other.second)
        return new_pair

class RSSVectorMyType:
    def __init__(self):
        self.pair = Pair()

    def random(self, shape):
        self.pair.random_pair(shape)


    def __getitem__(self, item):
        return self.pair[item]

    def __setitem__(self, key, value):
        self.pair[key] = value

    @property
    def first(self):
        return self.pair.first

    @property
    def second(self):
        return self.pair.second

    def __add__(self, other):
        new_rss = RSSVectorMyType()
        new_rss.pair = self.pair + other.pair
        return new_rss


def testMatMul(rows, common_dim, columns, iter, party):
    # make triple
    a = RSSVectorMyType()
    a.random(shape = [rows, common_dim])
    b = RSSVectorMyType()
    b.random(shape = [common_dim, columns])
    c = RSSVectorMyType()
    c.random(shape = [rows, columns])

    for i in range(iter):
        st = time.time()
        funcMatMul(a, b, c, rows, common_dim, columns, party)
        et = time.time()
        print("eval time: {} s".format(et-st))
        torch.cuda.empty_cache()


def matrixMultRSS(a, b):
    # /********************************* Triple For Loop *********************************/
    temp3 =  RingTensor.matmul(a.first, b.first) +  RingTensor.matmul(a.first, b.second) +  RingTensor.matmul(a.second, b.first)
    return temp3


def funcReconstruct3out3(a, b, party):
    if party.party_id == 0 or party.party_id == 1:
        party.send(2, a)
    if party.party_id == 2:
        tempA = party.receive(0)
        tempB = party.receive(1)
        tempA = tempA + a
        b = tempB + tempA
        party.send(0, b)
        party.send(1, b)
    if party.party_id == 0 or party.party_id == 1:
        b = party.receive(2)
    return b

def funcCheckMaliciousMatMul(a, b, c, rows, common_dim, columns):
    x, y, z = getTriplets(rows, common_dim, columns)


def getTriplets(rows, common_dim, columns):
    a = RSSVectorMyType()
    a.random(shape=[rows * common_dim])
    b = RSSVectorMyType()
    b.random(shape=[common_dim * columns])
    c = RSSVectorMyType()
    c.random(shape=[rows * columns])
    return a, b, c


def funcMatMul(a, b, c, rows, common_dim, columns, party):
    temp3 = matrixMultRSS(a, b)
    scale = 1000
    diffReconst = RingTensor.random([rows, columns])

    r = RSSVectorMyType()
    r.random(shape=[rows, columns])

    rPrime = RSSVectorMyType()
    rPrime.random(shape=[rows, columns])

    temp3 = temp3 - rPrime.first

    diffReconst = funcReconstruct3out3(temp3, diffReconst, party)

    diffReconst = diffReconst // scale
    # omit funcCheckMaliciousMatMul and dividePlain

    if party.party_id == 0:
        c.pair.first = r.first + diffReconst
        c.pair.second = r.second
    elif party.party_id == 1:
        c.pair.first = r.first
        c.pair.second = r.second
    else:
        c.pair.first = r.first
        c.pair.second = r.second + diffReconst

#
# #
# testMatMul(10, 5, 10, 1, None)