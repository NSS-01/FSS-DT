import time

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
    a.random(shape = [rows * common_dim])
    b = RSSVectorMyType()
    b.random(shape = [common_dim*columns])
    c = RSSVectorMyType()
    c.random(shape = [rows*columns])

    for i in range(iter):
        st = time.time()
        funcMatMul(a, b, c, rows, common_dim, columns, party)
        et = time.time()
        print("eval time: {} s".format(et-st))


def matrixMultRSS(a, b, temp3, rows, common_dim, columns):
    # /********************************* Triple For Loop *********************************/
    triple_a = RSSVectorMyType()
    triple_a.random(shape=[rows * common_dim])
    triple_b = RSSVectorMyType()
    triple_b.random(shape=[common_dim * columns])

    for i in range(rows):
        for j in range(common_dim):
            triple_a[i*common_dim + j] = a[i*common_dim + j]

    for i in range(common_dim):
        for j in range(columns):
            triple_b[i*columns + j] = b[i*columns + j]

    for i in range(rows):
        for j in range(columns):
            temp3[i * columns + j] = RingTensor(0)
            for k in range(common_dim):
                temp3[i*columns + j] += triple_a[i*common_dim + k].first * triple_b[k*columns + j].first + triple_a[i*common_dim + k].first * triple_b[k*columns + j].second + triple_a[i*common_dim + k].second * triple_b[k*columns + j].first

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
    return a, b

def funcCheckMaliciousMatMul(rows, common_dim, columns):
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
    final_size = rows * columns
    temp3 = RingTensor.random(shape=[final_size])
    temp3 = matrixMultRSS(a, b, temp3, rows, common_dim, columns)
    diffReconst = RingTensor.random([final_size])
    r = RSSVectorMyType()
    r.random(shape=[final_size])

    rPrime = RSSVectorMyType()
    rPrime.random(shape=[final_size])

    for i in range(final_size):
        temp3[i] = temp3[i] - rPrime[i].first

    temp3, diffReconst = funcReconstruct3out3(temp3, diffReconst, party)
    # omit funcCheckMaliciousMatMul and dividePlain

    if party.party_id == 0:
        for i in range(final_size):
            c[i].first = r[i].first + diffReconst[i]
            c[i].second = r[i].second
    elif party.party_id == 1:
        for i in range(final_size):
            c[i].first = r[i].first
            c[i].second = r[i].second
    else:
        for i in range(final_size):
            c[i].first = r[i].first
            c[i].second = r[i].second + diffReconst[i]

#
# #
# testMatMul(10, 5, 10, 1, None)