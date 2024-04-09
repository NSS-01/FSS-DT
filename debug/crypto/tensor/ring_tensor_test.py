import torch
from crypto.tensor.RingTensor import RingTensor


# test the sum function in RingTensor
def test_sum():
    print("test sum function in RingTensor")
    # create a tensor
    x = RingTensor(torch.tensor([[1, 2, 3], [4, 5, 6]]))
    print(x.sum(0))
    print()


# test the T function in RingTensor
def test_T():
    print("test T function in RingTensor")
    # create a tensor
    x = RingTensor(torch.tensor([[1, 2, 3], [4, 5, 6]]))
    print(x.T())
    print()


# test the add function in RingTensor
def test_add():
    print("test add function in RingTensor")
    x = RingTensor(torch.tensor([[1, 2, 3], [4, 5, 6]]))
    # add a RingTensor
    y = RingTensor(torch.tensor([[1, 2, 3], [4, 5, 6]]))
    print(x + y)
    print()


# test the sub function in RingTensor
def test_sub():
    print("test sub function in RingTensor")
    x = RingTensor(torch.tensor([[1, 2, 3], [4, 5, 6]]))
    # sub a RingTensor
    y = RingTensor(torch.tensor([[1, 2, 3], [4, 5, 6]]))
    print(x - y)
    print()


# test the mul function in RingTensor
def test_mul():
    print("test mul function in RingTensor")
    x = RingTensor(torch.tensor([[1, 2, 3], [4, 5, 6]]))
    # mul a RingTensor
    y = RingTensor(torch.tensor([[1, 2, 3], [4, 5, 6]]))
    print(x * y)
    print()


# test the neg function in RingTensor
def test_neg():
    print("test neg function in RingTensor")
    # create a tensor
    x = RingTensor(torch.tensor([[1, 2, 3], [4, 5, 6]]))
    print(-x)
    print()


# test random function in RingTensor
def test_random():
    print("test random function in RingTensor")
    # create a tensor
    x = RingTensor.random([2, 2])
    print(x)
    print()




# test save and load function in RingTensor
def test_save_load():
    print("test save and load function in RingTensor")
    # create a tensor
    x = RingTensor.random([2, 2])
    print(x)
    # save
    x.save('test.pt')
    # load
    y = RingTensor.load_from_file('test.pt')
    print(y)
    print()


def test_get_bit():
    print("test get_bit function in RingTensor")
    # create a tensor
    x = RingTensor.random([2, 2])
    print(x)
    # get_bit
    y = x.get_bit(0)
    print(y)
    print()

def test_repeate():
    print("test repeate func")
    x = RingTensor.random([3,2])
    print(x)
    y = x.repeat_interleave(4,1)
    print(y)
    print(y.shape)

def test_unsqueeze():
    print("test unsqueeze function in RingTensor")
    # create a tensor
    x = RingTensor.random([2, 2])
    print(x)
    # get_bit
    y = x.unsqueeze(0)
    print(y)
    print(y.shape)
    print()

def test_squeeze():
    print("test squeeze function in RingTensor")
    # create a tensor
    x = RingTensor.random([1, 2])
    print(x)
    # get_bit
    y = x.squeeze(0)
    print(y)
    print(y.shape)
    print()

def test_view():
    print("test view function in RingTensor")
    # create a tensor
    x = RingTensor.random([2, 2])
    print(x)
    # get_bit
    y = x.view(-1)
    print(y)
    print(y.shape)
    z = x.view(-1,1)
    print(z)
    print(z.shape)


def test_mod():
    print("test mod function in RingTensor")
    # create a tensor
    x = RingTensor.random([2, 2])
    print(x)
    # get_bit
    y = x % 2
    print(y)
    print(y.shape)

#
# test_sum()
# test_T()
# test_add()
# test_sub()
# test_mul()
# test_neg()
# test_random()
# test_save_load()
# test_get_bit()
# test_repeate()
# test_unsqueeze()
# test_squeeze()
# test_view()
test_mod()

