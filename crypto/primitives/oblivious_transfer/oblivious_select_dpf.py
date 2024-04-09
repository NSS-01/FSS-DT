from crypto.primitives.function_secret_sharing.verifiable_dpf import VerifiableDPF
from crypto.protocols.RSS_malicious_subprotocol.protocols import *


class ObliviousSelect(object):

    def __init__(self, party):
        self.self_rdx0 = None
        self.self_rdx1 = None
        self.k0 = None  # get from P+2
        self.k1 = None  # get from P+1
        self.rdx0 = None  # get from P+2
        self.rdx1 = None  # get from P+1
        self.party = party

    def preprocess(self, num_of_keys):
        party = self.party
        rdx = RingTensor.random([num_of_keys], dtype=party.dtype, scale=party.scale, device=DEVICE)
        k0, k1 = VerifiableDPF.gen(num_of_keys, rdx, RingTensor.convert_to_ring(1))
        rdx0, rdx1 = ArithmeticSecretSharing.share(rdx, 2)
        party.send_params_to((party.party_id + 1) % 3, k0.to_dic())
        party.send_params_to((party.party_id + 2) % 3, k1.to_dic())
        k0p2 = VerifiableDPFKey.dic_to_key(party.receive_params_dict_from((party.party_id + 2) % 3))
        k1p1 = VerifiableDPFKey.dic_to_key(party.receive_params_dict_from((party.party_id + 1) % 3))

        party.send_ring_tensor_to((party.party_id + 1) % 3, rdx0)
        party.send_ring_tensor_to((party.party_id + 2) % 3, rdx1)
        rdx0p2 = party.receive_ring_tensor_from((party.party_id + 2) % 3)
        rdx1p1 = party.receive_ring_tensor_from((party.party_id + 1) % 3)

        ObliviousSelect.check_keys_and_r()

        # self.self_rdx0 = rdx0.to('cpu')
        # self.self_rdx1 = rdx1.to('cpu')
        # self.k0 = k0p2.to('cpu')
        # self.k1 = k1p1.to('cpu')
        # self.rdx0 = rdx0p2.to('cpu')
        # self.rdx1 = rdx1p1.to('cpu')

        self.self_rdx0 = rdx0
        self.self_rdx1 = rdx1
        self.k0 = k0p2
        self.k1 = k1p1
        self.rdx0 = rdx0p2
        self.rdx1 = rdx1p1

    @staticmethod
    def check_keys_and_r():
        pass

    def selection(self, table: ReplicatedSecretSharing, idx: ReplicatedSecretSharing):
        idx = idx.view(-1, 1)
        table = table.unsqueeze(1)
        num = idx.shape[0]
        party = self.party

        if DEBUG:
            rdx_list = [ReplicatedSecretSharing([self.self_rdx1[0], self.self_rdx0[0]], party),
                        ReplicatedSecretSharing([RingTensor.convert_to_ring(0), self.rdx1[0]], party),
                        ReplicatedSecretSharing([self.rdx0[0], RingTensor.convert_to_ring(0)], party)
                        ]
        else:
            rdx_list = [ReplicatedSecretSharing([self.self_rdx1[:num], self.self_rdx0[:num]], party).view(-1, 1),
                        ReplicatedSecretSharing([RingTensor.convert_to_ring(0), self.rdx1[:num]], party).view(-1, 1),
                        ReplicatedSecretSharing([self.rdx0[:num], RingTensor.convert_to_ring(0)], party).view(-1, 1)
                        ]

        rdx_list = list_rotate(rdx_list, party.party_id)

        delta0 = idx - rdx_list[0]
        delta1 = idx - rdx_list[1]
        delta2 = idx - rdx_list[2]

        dt1_list = []
        dt2_list = []

        dt1_list.append(recon(delta1, 0))
        dt2_list.append(recon(delta2, 0))

        dt1_list.append(recon(delta2, 1))
        dt2_list.append(recon(delta0, 1))

        dt1_list.append(recon(delta0, 2))
        dt2_list.append(recon(delta1, 2))

        # dt1 = dt1_list[party.party_id].to(idx.device)
        # dt2 = dt2_list[party.party_id].to(idx.device)

        dt1 = dt1_list[party.party_id]
        dt2 = dt2_list[party.party_id]

        j = RingTensor(torch.arange(0, table.shape[-1], dtype=data_type, device=DEVICE),
                       idx.replicated_shared_tensor[0].dtype, idx.replicated_shared_tensor[0].scale, idx.device)

        # if DEBUG:
        #     v1, pi1 = VerifiableDPF.eval(j - dt1, self.k1[0].to(DEVICE), 1)
        #     v2, pi2 = VerifiableDPF.eval(j - dt2, self.k0[0].to(DEVICE), 0)
        # else:
        #     v1, pi1 = VerifiableDPF.eval(j - dt1, self.k1[:num].to(DEVICE), 1)
        #     v2, pi2 = VerifiableDPF.eval(j - dt2, self.k0[:num].to(DEVICE), 0)

        if DEBUG:
            v1, pi1 = VerifiableDPF.eval(j - dt1, self.k1[0], 1)
            v2, pi2 = VerifiableDPF.eval(j - dt2, self.k0[0], 0)
        else:
            v1, pi1 = VerifiableDPF.eval(j - dt1, self.k1[:num], 1)
            v2, pi2 = VerifiableDPF.eval(j - dt2, self.k0[:num], 0)

        # pi1 = RingTensor(pi1, idx.replicated_shared_tensor[0].dtype, idx.replicated_shared_tensor[0].scale, idx.device)
        # pi2 = RingTensor(pi2, idx.replicated_shared_tensor[0].dtype, idx.replicated_shared_tensor[0].scale, idx.device)

        # if party.party_id == 2:
        #     party.send_ring_tensor_to((party.party_id + 1) % 3, pi2)
        #     o_pi2 = party.receive_ring_tensor_from((party.party_id + 1) % 3)
        #     check_is_all_element_equal(pi2, o_pi2)
        #
        #     party.send_ring_tensor_to((party.party_id + 2) % 3, pi1)
        #     o_pi1 = party.receive_ring_tensor_from((party.party_id + 2) % 3)
        #     check_is_all_element_equal(pi1, o_pi1)
        # else:
        #     party.send_ring_tensor_to((party.party_id + 2) % 3, pi1)
        #     o_pi1 = party.receive_ring_tensor_from((party.party_id + 2) % 3)
        #     check_is_all_element_equal(pi1, o_pi1)
        #
        #     party.send_ring_tensor_to((party.party_id + 1) % 3, pi2)
        #     o_pi2 = party.receive_ring_tensor_from((party.party_id + 1) % 3)
        #     check_is_all_element_equal(pi2, o_pi2)

        v1 = RingTensor(v1, idx.replicated_shared_tensor[0].dtype, idx.replicated_shared_tensor[0].scale, idx.device)
        v2 = RingTensor(v2, idx.replicated_shared_tensor[0].dtype, idx.replicated_shared_tensor[0].scale, idx.device)

        res1 = (table.replicated_shared_tensor[0] * v1).sum(-1)
        res2 = (table.replicated_shared_tensor[1] * v2).sum(-1)

        res = res1 + res2

        return ReplicatedSecretSharing.reshare33(res, party)
