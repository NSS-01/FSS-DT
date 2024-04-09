import torch

from crypto.mpc.semi_honest_party import Party
from crypto.primitives.beaver.beaver import BeaverOfflineProvider
from crypto.primitives.arithmetic_secret_sharing.arithmetic_secret_sharing import ArithmeticSecretSharing

# test generate_triple_for_parties
if __name__ == '__main__':
    offline_provider = BeaverOfflineProvider()
    x = torch.rand((100, 784, 9))
    y = torch.rand((9, 32))
    offline_provider.gen_matrix_beaver_for_parties(x.shape, y.shape)
    x = torch.rand((100, 196, 288))
    y = torch.rand((288, 64))
    offline_provider.gen_matrix_beaver_for_parties(x.shape, y.shape)
    x = torch.rand((100, 49, 576))
    y = torch.rand((576, 128))
    offline_provider.gen_matrix_beaver_for_parties(x.shape, y.shape)
    x = torch.rand((100, 49, 1152))
    y = torch.rand((1152, 256))
    offline_provider.gen_matrix_beaver_for_parties(x.shape, y.shape)
    x = torch.rand((100, 49, 2304))
    y = torch.rand((2304, 256))
    offline_provider.gen_matrix_beaver_for_parties(x.shape, y.shape)
    x = torch.rand((100, 2304))
    y = torch.rand((2304, 1024))
    offline_provider.gen_matrix_beaver_for_parties(x.shape, y.shape)
    x = torch.rand((100, 1024))
    y = torch.rand((1024, 512))
    offline_provider.gen_matrix_beaver_for_parties(x.shape, y.shape)
    x = torch.rand((100, 512))
    y = torch.rand((512, 10))
    offline_provider.gen_matrix_beaver_for_parties(x.shape, y.shape)
    x = torch.rand((100, 32, 196, 4))
    y = torch.rand((4, 1))
    offline_provider.gen_matrix_beaver_for_parties(x.shape, y.shape)
    x = torch.rand((100, 64, 49, 4))
    y = torch.rand((4, 1))
    offline_provider.gen_matrix_beaver_for_parties(x.shape, y.shape)
    x = torch.rand((100, 256, 9, 4))
    y = torch.rand((4, 1))
    offline_provider.gen_matrix_beaver_for_parties(x.shape, y.shape)
    # test get_triples
    # party1 = Party(party_id=0)
    # offline_provider_1 = BeaverOfflineProvider()
    # m = torch.ones((2, 2))
    # n = torch.ones((2, 2))
    # a0, b0, c0 = offline_provider_1.get_mat_beaver(m.shape, n.shape, party1)
    # print(a0)

    # offline_provider_1.load_triples(party1, 2)
    #
    # party2 = Party(party_id=1)
    # offline_provider_2 = BeaverOfflineProvider()
    # a1, b1, c1 = offline_provider_2.get_mat_beaver(m.shape, n.shape, party2)
    # print(a1)
    # offline_provider_2.load_triples(party2, 2)
    #
    # a_sum = ArithmeticSecretSharing.restore_two_shares(a0, a1)
    # b_sum = ArithmeticSecretSharing.restore_two_shares(b0, b1)
    # c_sum = ArithmeticSecretSharing.restore_two_shares(c0, c1)
    # print(a_sum.shape)
    #
    # c_mul = a_sum @ b_sum
    # print(c_mul)
    # print(c_sum)
    #
    # a, b, c = offline_provider_1.get_triples_by_pointer(3)
    # print(a)
    # print(b)
    # print(c)
