import torch

from NssMPC.config import DEVICE
from NssMPC.crypto.protocols.arithmetic_secret_sharing.semi_honest_functional import b2a

from NssMPC import RingTensor
from application.neural_network.party import NeuralNetworkCS
from NssMPC.crypto.protocols.arithmetic_secret_sharing.semi_honest_functional.b2a import sonic_b2a
from NssMPC.crypto.protocols.look_up_table import LookUp
from debug.crypto.protocol.arithmetic_secret_sharing.basic_test.crypten_div import crypten_div
from debug.crypto.protocol.arithmetic_secret_sharing.basic_test.lut_pika import look_up_eval

client = NeuralNetworkCS(type='secure_version')
client.activate_by_gelu()
client.online()


def look_up_table_test(x):
    table = RingTensor.arange(0, 2 ** 12)
    k1 = client.receive()

    LookUp.eval(x, k1, table)


def look_up_table_test_pika(x):
    table = RingTensor.arange(0, 2 ** 12)
    k1 = client.receive()

    look_up_eval(x, k1, table)


def div_test(x, y):
    x / y


def div_test_crypten(x, y):
    crypten_div(x, y)


def b2a_test(x):
    z = b2a(x, client)


def sonic_b2a_test(x):
    z = sonic_b2a(x, client)


def batch_test(num_of_value):
    t = torch.randint(0, 2, [num_of_value], device=DEVICE)
    t_ring = RingTensor.convert_to_ring(t)

    x_1 = client.receive()
    y_1 = client.receive()

    # look_up_table_test(x_1)
    # torch.cuda.empty_cache()
    #
    # look_up_table_test_pika(x_1)
    # torch.cuda.empty_cache()

    div_test(x_1, y_1)
    torch.cuda.empty_cache()

    div_test_crypten(x_1, y_1)
    torch.cuda.empty_cache()

    b2a_test(t_ring)
    torch.cuda.empty_cache()

    sonic_b2a_test(t_ring)
    torch.cuda.empty_cache()


if __name__ == '__main__':
    for i in range(6):
        print(f"=================10^{i}==================")
        num_of_value = 10 ** i
        batch_test(num_of_value)
