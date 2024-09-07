from crypto.mpc.semi_honest_party import Party
from crypto.primitives.beaver.beaver import BeaverOfflineProvider
from crypto.primitives.arithmetic_secret_sharing.arithmetic_secret_sharing import ArithmeticSecretSharing

# test generate_triple_for_parties
if __name__ == '__main__':
    offline_provider = BeaverOfflineProvider()
    offline_provider.generate_triple_for_parties(num_of_triples=1000000, num_of_party=2)
    # test get_triples
    party1 = Party(party_id=0)
    offline_provider_1 = BeaverOfflineProvider()
    offline_provider_1.load_param(party1, 2)

    party2 = Party(party_id=1)
    offline_provider_2 = BeaverOfflineProvider()
    offline_provider_2.load_param(party2, 2)

    a_sum = ArithmeticSecretSharing.restore_two_shares(offline_provider_1.a_tensor, offline_provider_2.a_tensor)
    b_sum = ArithmeticSecretSharing.restore_two_shares(offline_provider_1.b_tensor, offline_provider_2.b_tensor)
    c_sum = ArithmeticSecretSharing.restore_two_shares(offline_provider_1.c_tensor, offline_provider_2.c_tensor)
    print(a_sum.shape)

    c_mul = a_sum * b_sum
    print(c_mul)
    print(c_sum)

    a, b, c = offline_provider_1.get_parameters(3)
    print(a)
    print(b)
    print(c)
