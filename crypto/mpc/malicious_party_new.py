from config.base_configs import BIT_LEN, PRG_TYPE, DEVICE, HALF_RING, SCALE, DTYPE
from common.random.prg import PRG
from config.mpc_configs import *

from crypto.primitives.arithmetic_secret_sharing.replicated_secret_sharing import ReplicatedSecretSharing
from crypto.primitives.arithmetic_secret_sharing.arithmetic_secret_sharing import ArithmeticSecretSharing
from crypto.primitives.oblivious_transfer.msb_with_os import MSBWithOS
from crypto.primitives.oblivious_transfer.oblivious_select_dpf import ObliviousSelect
from crypto.protocols.providers.auth_beaver_provider import AuthenticatedBeaverProvider
from crypto.protocols.providers.beaver_provider import BeaverProvider
from crypto.protocols.providers.key_provider import VSigmaKeyProvider

from crypto.tensor.RingTensor import RingTensor
from common.network.communicator import Communicator
from common.utils import *


class Malicious3PCParty:
    def __init__(self, party_id):
        """初始化PartyMPC的一个实例。
        :param party_id: int,此参与者的ID。
        """
        self.party_id = party_id
        self.communicator = Communicator()

        self.scale = SCALE
        self.dtype = DTYPE

        self.comm_rounds = 0
        self.comm_bytes = 0

        self.prg_seed_0 = None  # the prg seed hold by P_{i-1} and P_{i}
        self.prg_seed_1 = None  # the prg seed hold by P_{i} and P_{i+1}

        self.prg_0 = None
        self.prg_1 = None

        self.key_provider = None
        self.beaver_provider = None
        self.beaver_provider_2pc = None
        self.os_cmp = None
        self.os_dpf = None

    def online(self):
        self.key_provider = VSigmaKeyProvider(party=self)
        self.beaver_provider = BeaverProvider(party=self)
        self.os_cmp = MSBWithOS(party=self)
        self.os_dpf = ObliviousSelect(party=self)
        self.beaver_provider_2pc = AuthenticatedBeaverProvider(party=self)

        if self.party_id == 0:
            server_ip = (S0_ADDRESS, S0_PORT)
            client_ip = {1: (S0_ADDRESS, S0_client_port_to_S1), 2: (S0_ADDRESS, S0_client_port_to_S2)}
            target_server_ip = {1: (S1_ADDRESS, S1_PORT), 2: (S2_ADDRESS, S2_PORT)}
            client_mapping = {1: (S1_ADDRESS, S1_client_port_to_S0), 2: (S2_ADDRESS, S2_client_port_to_S0)}
        elif self.party_id == 1:
            server_ip = (S1_ADDRESS, S1_PORT)
            client_ip = {0: (S1_ADDRESS, S1_client_port_to_S0), 2: (S1_ADDRESS, S1_client_port_to_S2)}
            target_server_ip = {0: (S0_ADDRESS, S0_PORT), 2: (S2_ADDRESS, S2_PORT)}
            client_mapping = {0: (S0_ADDRESS, S0_client_port_to_S1), 2: (S2_ADDRESS, S2_client_port_to_S1)}
        elif self.party_id == 2:
            server_ip = (S2_ADDRESS, S2_PORT)
            client_ip = {0: (S2_ADDRESS, S2_client_port_to_S0), 1: (S2_ADDRESS, S2_client_port_to_S1)}
            target_server_ip = {0: (S0_ADDRESS, S0_PORT), 1: (S1_ADDRESS, S1_PORT)}
            client_mapping = {0: (S0_ADDRESS, S0_client_port_to_S2), 1: (S1_ADDRESS, S1_client_port_to_S2)}
        else:
            raise ValueError("Invalid party id.")

        self.communicator.set_address(server_ip[0], server_ip[1])
        self.communicator.start_tcp_process(client_ip, target_server_ip, client_mapping)
        self.communicator.connect()

    def generate_prg_seed(self):
        # generate prg seed 0
        prg_seed_0 = torch.randint(-10, 10, (1,))
        # send prg seed 0 to P_{i-1}
        self.send_torch_tensor_to((self.party_id + 2) % 3, prg_seed_0)
        self.prg_seed_0 = prg_seed_0.item()
        # receive prg seed 1 from P_{i+1}
        self.prg_seed_1 = self.receive_torch_tensor_from((self.party_id + 1) % 3).item()

        self.prg_0 = PRG(PRG_TYPE, device=DEVICE)
        self.prg_0.set_seed(self.prg_seed_0)
        self.prg_1 = PRG(PRG_TYPE, device=DEVICE)
        self.prg_1.set_seed(self.prg_seed_1)

    def set_scale(self, scale):
        self.scale = scale

    def set_dtype(self, dtype):
        self.dtype = dtype

    def set_ver_sigma_key_provider(self):
        self.key_provider = VSigmaKeyProvider(party=self)
        self.key_provider.load_keys(3)

    def close(self):
        self.communicator.close()

    def send_torch_tensor_to(self, target_id, tensor):
        self.communicator.send_msg_to(target_id, tensor)

    def send_ring_tensor_to(self, target_id, x):
        self.communicator.send_msg_to(target_id, x.tensor)

    def send_rss_to(self, target_id, x):
        self.communicator.send_msg_to(target_id, torch.cat([x.replicated_shared_tensor[0].tensor.unsqueeze(0),
                                                            x.replicated_shared_tensor[1].tensor.unsqueeze(0)]))

    def send_ass_to(self, target_id, share):
        self.communicator.send_msg_to(target_id, share.ring_tensor.tensor)

    def send_params_to(self, target_id, keys):
        self.communicator.send_msg_to(target_id, keys if isinstance(keys, dict) else keys.to_dic())

    def receive_torch_tensor_from(self, target_id):
        return self.communicator.recv_from(target_id)

    def receive_ring_tensor_from(self, target_id):
        tensor = self.communicator.recv_from(target_id)
        return RingTensor.load_from_value(tensor, self.dtype)

    def receive_ass_from(self, target_id):
        tensor = self.communicator.recv_from(target_id)
        return ArithmeticSecretSharing.load_from_ring_tensor(tensor, self)

    def receive_rss_from(self, target_id):
        tensor = self.communicator.recv_from(target_id)
        ring_tensor0 = RingTensor.load_from_value(tensor[0], self.dtype)
        ring_tensor1 = RingTensor.load_from_value(tensor[1], self.dtype)
        return ReplicatedSecretSharing.load_from_ring_tensor([ring_tensor0, ring_tensor1], self)

    def receive_params_dict_from(self, target_id):
        param_dict = self.communicator.recv_from(target_id)
        if not isinstance(param_dict, dict):
            raise TypeError("The type of param_dict must be dict.")
        # TODO: self.comm_bytes
        return param_dict

    def print_communication_stats(self):
        """ 打印通信统计信息

        :return:
        """
        print("Communication rounds(include send and recv): ", self.comm_rounds)
        print("Communication bytes(include send and recv): ", self.comm_bytes, "b")
