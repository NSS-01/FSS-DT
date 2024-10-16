import torch
from NssMPC.crypto.protocols.arithmetic_secret_sharing.semi_honest_functional import b2a

from NssMPC import RingTensor, ArithmeticSecretSharing
from application.neural_network.party import NeuralNetworkCS
from NssMPC.common.utils import get_time, comm_count
from NssMPC.config import DEVICE
from NssMPC.crypto.aux_parameter.look_up_table_keys.lut_key import LookUpKey
from NssMPC.crypto.protocols.arithmetic_secret_sharing.semi_honest_functional.b2a import sonic_b2a
from NssMPC.crypto.protocols.look_up_table import LookUp
from debug.crypto.protocol.arithmetic_secret_sharing.basic_test.crypten_div import crypten_div
from debug.crypto.protocol.arithmetic_secret_sharing.basic_test.lut_pika import look_up_gen, look_up_eval

server = NeuralNetworkCS(type='server')
server.activate_by_gelu()
server.online()


def look_up_table_test(x):
    table = RingTensor.arange(0, 2 ** 12)
    k0, k1 = LookUpKey.gen(1, 0, 2 ** 12)
    server.send(k1)

    comm_count(server.communicator, get_time, LookUp.eval, x, k0, table)


def look_up_table_test_pika(x):
    table = RingTensor.arange(0, 2 ** 12)
    k0, k1 = look_up_gen(1, 0, 2 ** 12)
    server.send(k1)

    comm_count(server.communicator, get_time, look_up_eval, x, k0, table)


def div_test(x, y):
    x / y


def div_test_crypten(x, y):
    crypten_div(x, y)


def b2a_test(x):
    b2a(x, server)


def sonic_b2a_test(x):
    sonic_b2a(x, server)


def batch_test(num_of_value):
    x = torch.rand([num_of_value], device=DEVICE)
    y = torch.rand([num_of_value], device=DEVICE)
    t = torch.randint(0, 2, [num_of_value], device=DEVICE)

    x_ring = RingTensor.convert_to_ring(x)
    y_ring = RingTensor.convert_to_ring(y)
    t_ring = RingTensor.convert_to_ring(t)

    x_0, x_1 = ArithmeticSecretSharing.share(x_ring, 2)
    server.send(x_1)
    x_0.party = server

    y_0, y_1 = ArithmeticSecretSharing.share(y_ring, 2)
    server.send(y_1)
    y_0.party = server

    # print("===============Our LUT====================")
    # look_up_table_test(x_0)
    # torch.cuda.empty_cache()
    # print("===================================")

    # print("================Pika LUT===================")
    # look_up_table_test_pika(x_0)
    # torch.cuda.empty_cache()
    # print("===================================")
    #
    # print("================Our Div===================")
    # comm_count(server.communicator, get_time, div_test, x_0, y_0)
    # torch.cuda.empty_cache()
    # print("===================================")
    #
    # print("================Crypten Div===================")
    # comm_count(server.communicator, get_time, div_test_crypten, x_0, y_0)
    # torch.cuda.empty_cache()
    # print("===================================")
    #
    # print("=================Our b2a==================")
    # comm_count(server.communicator, get_time, b2a_test, t_ring)
    # torch.cuda.empty_cache()
    # print("===================================")
    #
    # print("==================Sonic b2a=================")
    # comm_count(server.communicator, get_time, sonic_b2a_test, t_ring)
    # torch.cuda.empty_cache()
    # print("===================================")


if __name__ == '__main__':
    for i in range(6):
        print(f"=================10^{i}==================")
        num_of_value = 500 * 500
        batch_test(num_of_value)
