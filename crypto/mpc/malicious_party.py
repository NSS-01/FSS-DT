from common.network.async_tcp import *
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
import torch


class Malicious3PCParty:
    def __init__(self, party_id):
        """初始化PartyMPC的一个实例。
        :param party_id: int,此参与者的ID。
        """
        self.party_id = party_id
        self.server = None
        self.client2other = {}
        self.target_client_mapping = {}

        self.scale = SCALE
        self.dtype = DTYPE

        self.comm_rounds = 0
        self.comm_bytes = 0
        self.comm_tensors = 0

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
            self.start_server(S0_ADDRESS, S0_PORT)
            mapping = {1: (S1_ADDRESS, S1_client_port_to_S0), 2: (S2_ADDRESS, S2_client_port_to_S0)}
            self.set_target_client_mapping(mapping)
            self.connect_to_other(1, S0_ADDRESS, S0_client_port_to_S1, S1_ADDRESS, S1_PORT)
            self.connect_to_other(2, S0_ADDRESS, S0_client_port_to_S2, S2_ADDRESS, S2_PORT)
            self.check_all_parties_online()

        elif self.party_id == 1:
            self.start_server(S1_ADDRESS, S1_PORT)
            mapping = {0: (S0_ADDRESS, S0_client_port_to_S1), 2: (S2_ADDRESS, S2_client_port_to_S1)}
            self.set_target_client_mapping(mapping)
            self.connect_to_other(0, S1_ADDRESS, S1_client_port_to_S0, S0_ADDRESS, S0_PORT)
            self.connect_to_other(2, S1_ADDRESS, S1_client_port_to_S2, S2_ADDRESS, S2_PORT)
            self.check_all_parties_online()

        elif self.party_id == 2:
            self.start_server(S2_ADDRESS, S2_PORT)
            mapping = {0: (S0_ADDRESS, S0_client_port_to_S2), 1: (S1_ADDRESS, S1_client_port_to_S2)}
            self.set_target_client_mapping(mapping)
            self.connect_to_other(0, S2_ADDRESS, S2_client_port_to_S0, S0_ADDRESS, S0_PORT)
            self.connect_to_other(1, S2_ADDRESS, S2_client_port_to_S1, S1_ADDRESS, S1_PORT)
            self.check_all_parties_online()

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

    def start_server(self, address, port):
        self.server = TCPServer(address, port)
        self.server.run()

    def connect_to_other(self, target_id, self_address, self_port, target_address, target_port):
        client = TCPClient(self_address, self_port)
        client.connect_to_with_retry(target_address, target_port)
        self.client2other[target_id] = client

    def close_all(self):
        for client in self.client2other.values():
            client.close()
        self.server.server_socket.close()

    def send_torch_tensor_to(self, target_id, tensor):
        self.comm_rounds += 1
        socket = self.client2other.get(target_id)
        socket.send_serializable_item(tensor)
        self.comm_bytes += tensor.element_size() * BIT_LEN

    def send_ring_tensor_to(self, target_id, x):
        self.comm_rounds += 1
        self.comm_tensors += x.shape.numel()
        socket = self.client2other.get(target_id)
        socket.send_serializable_item(x.tensor)
        self.comm_bytes += x.tensor.element_size() * BIT_LEN

    def send_rss_to(self, target_id, x):
        self.comm_rounds += 1
        socket = self.client2other.get(target_id)
        socket.send_serializable_item(x.replicated_shared_tensor[0].tensor)
        socket.send_serializable_item(x.replicated_shared_tensor[1].tensor)
        self.comm_bytes += x.replicated_shared_tensor[0].tensor.element_size() * BIT_LEN * 2

    def send_ass_to(self, target_id, share):
        self.comm_rounds += 1
        socket = self.client2other.get(target_id)
        socket.send_serializable_item(share.ring_tensor.tensor)
        self.comm_bytes += share.ring_tensor.tensor.element_size() * BIT_LEN

    def send_params_to(self, target_id, keys):
        self.comm_rounds += 1
        socket = self.client2other.get(target_id)
        if isinstance(keys, dict):
            socket.send_serializable_item(keys)
        else:
            socket.send_serializable_item(keys.to_dic())
        # TODO: self.comm_bytes

    def receive_torch_tensor_from(self, target_id):
        self.comm_rounds += 1
        tensor = self.server.receive_serializable_item_from(self.target_client_mapping.get(target_id))
        self.comm_bytes += tensor.element_size() * BIT_LEN
        return tensor

    def receive_ring_tensor_from(self, target_id):
        self.comm_rounds += 1
        tensor = self.server.receive_serializable_item_from(self.target_client_mapping.get(target_id))
        self.comm_bytes += tensor.element_size() * BIT_LEN
        return RingTensor.load_from_value(tensor, self.dtype, self.scale)

    def receive_ass_from(self, target_id):
        self.comm_rounds += 1
        tensor = self.server.receive_serializable_item_from(self.target_client_mapping.get(target_id))
        self.comm_bytes += tensor.element_size() * BIT_LEN
        return ArithmeticSecretSharing.load_from_ring_tensor(tensor, self)

    def receive_rss_from(self, target_id):
        self.comm_rounds += 1
        tensor0 = self.server.receive_serializable_item_from(self.target_client_mapping.get(target_id))
        tensor1 = self.server.receive_serializable_item_from(self.target_client_mapping.get(target_id))
        self.comm_bytes += tensor0.element_size() * BIT_LEN * 2
        RT1 = RingTensor.load_from_value(tensor0, self.dtype, self.scale)
        RT2 = RingTensor.load_from_value(tensor1, self.dtype, self.scale)
        return ReplicatedSecretSharing.load_from_ring_tensor([RT1, RT2], self)

    def receive_params_dict_from(self, target_id):
        self.comm_rounds += 1
        param_dict = self.server.receive_serializable_item_from(self.target_client_mapping.get(target_id))
        if not isinstance(param_dict, dict):
            raise TypeError("The type of param_dict must be dict.")
        # TODO: self.comm_bytes
        return param_dict

    def check_all_parties_online(self):
        """ 检查所有的参与者是否在线。
        :return:
        """
        while self.server.connect_number < (3 - 1):
            pass

    def set_target_client_mapping(self, mapping):
        self.target_client_mapping = mapping

    def print_communication_stats(self):
        """ 打印通信统计信息

        :return:
        """
        print("Communication rounds(include send and recv): ", self.comm_rounds)
        print("Communication bytes(include send and recv): ", self.comm_bytes, "b")
