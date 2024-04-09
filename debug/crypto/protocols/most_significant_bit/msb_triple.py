from crypto.protocols.most_significant_bit.msb_triple_provider import MSBProvider

provider = MSBProvider()

bit_len = 1000000

provider.gen_and_save(bit_len)
