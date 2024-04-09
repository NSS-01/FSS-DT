import torch

from application.clustering.PLE_DBSCAN.protocols.edc_parallelization import RandomRSSProvider
from common.network.async_tcp import *
from common.random.prg import PRG
from common.utils import *
from config.base_configs import BIT_LEN, GE_TYPE, DEBUG, PRG_TYPE, DEVICE
from crypto.primitives.arithmetic_secret_sharing.improved_secret_sharing import ImprovedSecretSharing
from crypto.primitives.auxiliary_parameter.buffer_thread import BufferThread
from crypto.primitives.auxiliary_parameter.param_buffer import ParamBuffer
from crypto.primitives.function_secret_sharing.dicf import DICFKey
from crypto.primitives.function_secret_sharing.p_dicf import ParityDICFKey
from crypto.primitives.msb.msb_triples import MSBTriples
from crypto.protocols.comparison.cmp_sigma import CMPSigmaKey
from crypto.protocols.exp.pos_exp import PosExpKey
from crypto.protocols.exp.neg_exp import NegExpKey
from crypto.protocols.truncate.tr_crypten import Wrap
from crypto.tensor.ArithmeticSharedRingTensor import ArithmeticSharedRingTensor
# from crypto.tensor.ReplicatedSharedRingTensor import ReplicatedSharedRingTensor
from crypto.tensor.RingTensor import RingTensor


class Party(object):
    """ 代表一个参与方的类。
    每个Party对象都有一个唯一的party_id来标识它。
    Attributes:
        party_id (int): 代表参与方的唯一标识符。
    """

    def __init__(self, party_id):
        """ 初始化一个参与方对象
        :param party_id:int,用于标识参与方对象的唯一标识符
        """
        self.party_id = party_id


class SemiHonestCS(Party):
    """ 半诚实设置下的客户端-服务器模型

    此类支持两种类型：客户端或服务器。每种类型的实体都有其特定的party_id：客户端为1，服务器为0。同时，每个实体都有与之关联的TCP连接参数，以及默认的数据类型和缩放。

    :Attributes:
        type (str): 可以是'client'或'server'，表示实体的类型。
        tcp_address (str): 用于TCP连接的地址。
        tcp_port (int): 用于TCP连接的端口。
        tcp (TCPObject): 表示TCP连接的对象。
        scale (int): 用于数据的缩放。
        dtype (str): 用于数据的数据类型，如'int'。
        beaver_provider: Beaver三元组的提供者，用于安全计算。
    """

    # Client-Server model in semi-honest setting
    def __init__(self, type='client'):
        assert type in ('client', 'server'), "type must be 'client' or 'server'"
        self.type = type
        party_id = 1 if type == 'server' else 0
        super(SemiHonestCS, self).__init__(party_id)
        self.server = None
        self.client = None
        self.other_client_address = None
        self.comm_rounds = {'send': 0, 'recv': 0}
        self.comm_bytes = {'send': 0, 'recv': 0}

        self.beaver_provider = None
        self.wrap_provider = None
        self.compare_key_provider = None
        self.neg_exp_provider = None
        self.pos_exp_provider = None
        self.exp_provider = None

        self.beaver_provider_thread = None
        self.wrap_provider_thread = None
        self.compare_key_provider_thread = None
        self.neg_exp_provider_thread = None
        self.pos_exp_provider_thread = None
        self.exp_provider_thread = None

    def set_server(self, address):
        assert isinstance(address, tuple)
        self.server = TCPServer(address[0], address[1])

    def set_client(self, address):
        assert isinstance(address, tuple)
        self.client = TCPClient(address[0], address[1])

    def set_client_address(self, address):
        self.other_client_address = address

    def connect(self, self_server_socket, self_client_socket, other_server_socket, other_client_socket):
        assert isinstance(self_server_socket, tuple)
        assert isinstance(self_client_socket, tuple)
        assert isinstance(other_server_socket, tuple)
        assert isinstance(other_client_socket, tuple)

        self.set_server(self_server_socket)
        self.set_client(self_client_socket)
        self.server.run()
        self.client.connect_to_with_retry(other_server_socket[0], other_server_socket[1])
        self.check_all_parties_online()
        self.set_client_address(other_client_socket)
        if self.beaver_provider_thread is not None:
            self.beaver_provider_thread.start()

    def check_all_parties_online(self):
        """ 检查所有的参与者是否在线。
        :return:
        """
        while self.server.connect_number < 1:
            pass

    def set_beaver_provider(self, beaver_provider):
        self.beaver_provider = beaver_provider
        if not DEBUG:
            self.beaver_provider_thread = BufferThread(self.beaver_provider, self)

    def set_wrap_provider(self):
        self.wrap_provider = ParamBuffer(Wrap, self)
        self.wrap_provider.load_param()
        if not DEBUG:
            self.wrap_provider_thread = BufferThread(self.wrap_provider, self)

    def set_neg_exp_provider(self):
        self.neg_exp_provider = ParamBuffer(NegExpKey, self)
        self.neg_exp_provider.load_param()
        if not DEBUG:
            self.neg_exp_provider_thread = BufferThread(self.neg_exp_provider, self)

    def set_pos_exp_provider(self):
        self.pos_exp_provider = ParamBuffer(PosExpKey, self)
        self.pos_exp_provider.load_param()
        if not DEBUG:
            self.pos_exp_provider_thread = BufferThread(self.pos_exp_provider, self)

    def set_exp_provider(self):
        self.exp_provider = ParamBuffer(PosExpKey, self)
        self.exp_provider.load_param()
        if not DEBUG:
            self.exp_provider_thread = BufferThread(self.exp_provider, self)

    def set_compare_key_provider(self, num_of_parties=2):
        if GE_TYPE == 'FSS':
            self.compare_key_provider = ParamBuffer(DICFKey, self)
        elif GE_TYPE == 'MSB':
            self.compare_key_provider = ParamBuffer(MSBTriples, self)
        elif GE_TYPE == 'GROTTO':
            self.compare_key_provider = ParamBuffer(ParityDICFKey, self)
        elif GE_TYPE == "SIGMA":
            self.compare_key_provider = ParamBuffer(CMPSigmaKey, self)
        else:
            raise Exception("Wrong GE_TYPE")
        self.compare_key_provider.load_param()
        if not DEBUG:
            self.compare_key_provider_thread = BufferThread(self.compare_key_provider, self)

    def set_beaver_provider_thread(self, beaver_provider_thread):
        self.beaver_provider_thread = beaver_provider_thread

    def send(self, x):
        self.comm_rounds['send'] += 1
        self.comm_bytes['send'] += count_bytes(x)
        self.client.send_serializable_item(x)

    def receive(self):
        ret = self.server.receive_serializable_item_from(self.other_client_address)
        self.comm_rounds['recv'] += 1
        self.comm_bytes['recv'] += count_bytes(ret)
        if isinstance(ret, ArithmeticSharedRingTensor):
            ret.party = self
        return ret

    def wait(self):
        """
        保证两方通信同步
        :return:
        """
        self.send(torch.tensor(1))
        self.receive()

    def close(self):
        """ 关闭TCP连接。

        :return:
        """
        self.client.close()
        self.server.close_all()
        if self.beaver_provider_thread is not None:
            self.beaver_provider_thread.join()
        import os
        os.kill(os.getpid(), 0)
        print(f"Communication costs:\n\tsend rounds: {self.comm_rounds['send']}\t\t"
              f"send bytes: {bytes_convert(self.comm_bytes['send'])}.")
        print(f"\trecv rounds: {self.comm_rounds['recv']}\t\t"
              f"recv bytes: {bytes_convert(self.comm_bytes['recv'])}.")


class SemiHonestMPCParty(Party):
    """
    多方计算的参与者。

    此类包含多个功能，允许参与者与其他参与者进行通信，发送和接收数据，并加载和使用密钥和掩码。

    Attributes:
        party_id (int): 此参与者的ID。
        parties_num (int): 参与多方计算的总参与者数量。
        server (TCPServer): 用于接收来自其他参与者的数据的服务器。
        client2other (dict): 字典，其键是其他参与者的ID，其值是TCPClient对象，用于向其他参与者发送数据。
        target_client_mapping (dict): 用于将其他参与者的ID映射到特定的客户端对象。
        scale (int): 用于数据的缩放。
        dtype (str): 用于数据的数据类型，如'int'。
        comm_rounds (int): 记录通信轮次的计数器。
        comm_bytes (int): 记录发送和接收的字节数的计数器。
        pre_generated_mask: 预生成的掩码，用于数据隐私。
        dicf_0, dicf_1, dicf_from_next, dicf_from_pre: DICF提供者的对象，用于安全计算。
    """

    def __init__(self, id, parties_num, scale=1, dtype='int'):
        """初始化PartyMPC的一个实例。

        :param id: int,此参与者的ID。
        :param parties_num: int,参与多方计算的总参与者数量。
        :param scale: int, optional,用于数据的缩放。默认为1。
        :param dtype: str, optional,用于数据的数据类型。默认为'int'。
        """
        super(SemiHonestMPCParty, self).__init__(id)
        self.party_id = id
        self.parties_num = parties_num
        self.server = None
        self.client2other = {}  # 先用字典试试
        self.target_client_mapping = {}

        # 先固定scale和dtype， 后面再考虑mix mode
        self.scale = scale
        self.dtype = dtype

        self.comm_rounds = 0
        self.comm_bytes = 0

        self.pre_generated_mask = None

        self.compare_key_provider = None

        self.rss_provider = None

        self.prg_seed_0 = None
        self.prg_seed_1 = None

        self.prg_0 = None
        self.prg_1 = None

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

    def load_pre_generated_mask(self, file_path):
        """从文件中加载预生成的掩码。

        :param file_path:
        :return:
        """
        pre_generated_mask_0 = RingTensor.load_from_file(file_path + 'mask_0.pth')
        pre_generated_mask_1 = RingTensor.load_from_file(file_path + 'mask_1.pth')
        self.pre_generated_mask = ReplicatedSharedRingTensor(
            replicated_shared_tensor=[pre_generated_mask_0, pre_generated_mask_1], party=self)

    def get_pre_generated_mask(self, num_of_samples):
        """从预生成的掩码中获取指定样本数的掩码。

        从self.pre_generated_mask中取得掩码的两部分，并存储在r1和r2中。
        然后，它创建了两个包含num_of_samples数量的1的向量，并将这些1的向量与r1和r2相乘，得到o1和o2。
        使用RingTensor.load_from_value方法将这两个新的向量转换为RingTensor类型。
        最后，它返回一个新的ReplicatedSecretSharing对象，其中包含这两个新创建的RingTensor对象。

        :param num_of_samples:int,给定的样本数量
        :return:包含两个RingTensor对象的复制秘密共享
        """
        # TODO:这里改成单一元素复制。
        r1 = self.pre_generated_mask.replicated_shared_tensor[0].tensor[1]
        r2 = self.pre_generated_mask.replicated_shared_tensor[1].tensor[1]
        # print("r1: ", r1)
        # print("r2: ", r2)

        o1 = torch.ones(num_of_samples, dtype=torch.int64) * r1
        o2 = torch.ones(num_of_samples, dtype=torch.int64) * r2

        o1 = RingTensor.load_from_value(o1)
        o2 = RingTensor.load_from_value(o2)

        return ReplicatedSharedRingTensor(replicated_shared_tensor=[o1, o2], party=self)

        # return self.pre_generated_mask[:num_of_samples]

    def set_scale(self, scale):
        self.scale = scale

    def set_dtype(self, dtype):
        self.dtype = dtype

    def set_parity_dicf_key_provider(self):
        """
        设置大小比较三元组/密钥提供器
        """
        self.compare_key_provider = ParamBuffer(ParityDICFKey, party=self)
        self.compare_key_provider.load_param()

    def set_cmp_sigma_key_provider(self):
        self.compare_key_provider = ParamBuffer(CMPSigmaKey, party=self)

    def set_compare_key_provider(self, key_provider=None):
        if key_provider is None and GE_TYPE == 'FSS':
            self.compare_key_provider = ParamBuffer(DICFKey, party=self)
            self.compare_key_provider.load_param()
        elif key_provider is None:
            self.set_cmp_sigma_key_provider()
        else:
            self.compare_key_provider = key_provider

    def start_server(self, address, port):
        """ 启动服务器以接收来自其他参与者的数据。

        :param address:服务器地址
        :param port:服务器端口
        :return:
        """
        self.server = TCPServer(address, port)
        self.server.run()

    def connect_to_other(self, id, self_address, self_port, target_address, target_port):
        """ 连接到另一个指定的参与者。

        :param id:参与者ID
        :param self_address:自身地址
        :param self_port:自身端口
        :param target_address:目标参与者地址
        :param target_port:目标参与者端口
        :return:
        """
        client = TCPClient(self_address, self_port)
        # client.connect_to(target_address, target_port)
        client.connect_to_with_retry(target_address, target_port)
        self.client2other[id] = client

    def close_all(self):
        # 关闭自身所有的socket连接
        for client in self.client2other.values():
            client.close()
        self.server.server_socket.close()

    def send_torch_tensor_to(self, id, tensor):
        """ 向指定的参与者发送torch张量。

        :param id:参与者ID
        :param tensor:发送的张量
        :return:
        """
        self.comm_rounds += 1
        socket = self.client2other.get(id)
        socket.send_serializable_item(tensor)
        self.comm_bytes += tensor.element_size() * BIT_LEN

    def send_ring_tensor_to(self, id, x):
        """ 向指定参与者返送环上的张量
        :param id:
        :param x:
        :return:
        """
        self.comm_rounds += 1
        socket = self.client2other.get(id)
        socket.send_serializable_item(x.tensor)
        self.comm_bytes += x.tensor.element_size() * BIT_LEN

    def send_rss_to(self, id, x):
        """ 发送RSS(Replicated Secret Sharing)给指定ID的客户端

        :param id: 客户端ID
        :param x: 发送内容
        :return:
        """
        # print("send rss to" + str(id))
        self.comm_rounds += 1
        socket = self.client2other.get(id)
        socket.send_serializable_item(x)
        self.comm_bytes += x.tensor.element_size() * BIT_LEN * 2

    def send_iss_to(self, id, x):
        """ 发送ISS(Improved Secret Sharing)给指定ID的客户端

        :param id:
        :param x:
        :return:
        """
        # print("send iss to" + str(id))
        self.comm_rounds += 1
        socket = self.client2other.get(id)
        socket.send_serializable_item(x.public.tensor)
        socket.send_serializable_item(x.replicated_shared_tensor.replicated_shared_tensor[0].tensor)
        socket.send_serializable_item(x.replicated_shared_tensor.replicated_shared_tensor[1].tensor)
        self.comm_bytes += x.replicated_shared_tensor.replicated_shared_tensor[0].tensor.element_size() * BIT_LEN * 3

    def send_share_to(self, id, share):
        """ 发送密钥给指定ID的客户端

        :param id:
        :param share:
        :return:
        """
        self.comm_rounds += 1
        socket = self.client2other.get(id)
        socket.send_serializable_item(share.ring_tensor.tensor)
        self.comm_bytes += share.ring_tensor.tensor.element_size() * BIT_LEN

    def send_params_to(self, id, keys):
        self.comm_rounds += 1
        socket = self.client2other.get(id)
        if isinstance(keys, dict):
            socket.send_serializable_item(keys)
        else:
            socket.send_serializable_item(keys.to_dic())
        # TODO: self.comm_bytes

    def receive_torch_tensor_from(self, id):
        """ 从指定ID的客户端接收torch张量

        :param id:
        :return:
        """
        self.comm_rounds += 1
        tensor = self.server.receive_serializable_item_from(self.target_client_mapping.get(id))
        self.comm_bytes += tensor.element_size() * BIT_LEN
        return tensor

    def receive_ring_tensor_from(self, id):
        """ 从指定ID的客户端接收Ring张量

        :param id:
        :return:
        """
        self.comm_rounds += 1
        tensor = self.server.receive_serializable_item_from(self.target_client_mapping.get(id))
        self.comm_bytes += tensor.element_size() * BIT_LEN
        # print("receive ring tensor from " + str(id))
        # print(tensor.shape)
        return RingTensor.load_from_value(tensor, self.dtype)

    def receive_shares_from(self, id):
        """ 从指定ID的客户端接收密钥共享

        :param id:
        :return:
        """
        self.comm_rounds += 1
        tensor = self.server.receive_serializable_item_from(self.target_client_mapping.get(id))
        self.comm_bytes += tensor.element_size() * BIT_LEN
        return ArithmeticSharedRingTensor(tensor, self)

    def receive_rss_from(self, id):
        """ 从指定ID的客户端接收RSS

        :param id:
        :return:
        """
        self.comm_rounds += 1
        tensors = self.server.receive_serializable_item_from(self.target_client_mapping.get(id))
        self.comm_bytes += tensors.tensor.element_size() * BIT_LEN * 2
        tensors.party = self
        return tensors

    def receive_iss_from(self, id):
        """ 从指定ID的客户端接收ISS

        :param id:
        :return:
        """
        self.comm_rounds += 1
        public = self.server.receive_serializable_item_from(self.target_client_mapping.get(id))
        tensor0 = self.server.receive_serializable_item_from(self.target_client_mapping.get(id))
        tensor1 = self.server.receive_serializable_item_from(self.target_client_mapping.get(id))
        self.comm_bytes += tensor0.element_size() * BIT_LEN * 3
        RT_b = RingTensor.load_from_value(public, self.dtype)
        RT1 = RingTensor.load_from_value(tensor0, self.dtype)
        RT2 = RingTensor.load_from_value(tensor1, self.dtype)
        rss = ReplicatedSharedRingTensor.load_from_ring_tensor([RT1, RT2], self)
        iss = ImprovedSecretSharing(RT_b, rss, self)
        return iss

    def receive_params_dict_from(self, id):
        self.comm_rounds += 1
        param_dict = self.server.receive_serializable_item_from(self.target_client_mapping.get(id))
        if not isinstance(param_dict, dict):
            raise TypeError("The type of param_dict must be dict.", type(param_dict))
        # TODO: self.comm_bytes
        return param_dict

    def check_all_parties_online(self):
        """ 检查所有的参与者是否在线。
        :return:
        """
        while self.server.connect_number < (self.parties_num - 1):
            pass

    def set_target_client_mapping(self, mapping):
        """ 设置客户端映射

        :param mapping:
        :return:
        """
        self.target_client_mapping = mapping

    def print_communication_stats(self):
        """ 打印通信统计信息

        :return:
        """
        print("Communication rounds(include send and recv): ", self.comm_rounds)
        print("Communication bytes(include send and recv): ", self.comm_bytes, "b")

    def set_rss_provider(self):
        self.rss_provider = RandomRSSProvider()

    def load_rss_provider(self, num_of_rss):
        self.rss_provider.load(num_of_rss, self)
