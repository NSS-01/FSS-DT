import struct
import time
from multiprocessing import Pipe
from threading import Thread, Lock

from common.network.mt_tcp import *
from config.base_configs import SOCKET_NUM
from config.mpc_configs import *
from common.utils import *


class TCPProcess(object):
    def __init__(self, server_ip, client_ip, target_server_ip, client_mapping, pipe=Pipe(True), lock=Lock(),
                 socket_num=SOCKET_NUM):
        self.out_pipe, self.in_pipe = pipe
        self.lock = lock
        self.socket_num = socket_num
        self.sub_p = None
        self.server_ip = server_ip
        self.client_ip = client_ip
        self.target_server_ip = target_server_ip
        self.client_mapping = client_mapping
        self.world_size = len(client_ip) + 1

    def start(self):
        self.sub_p = Thread(target=tcp_process, args=(
            (self.out_pipe, self.in_pipe), self.world_size, self.socket_num, self.server_ip, self.client_ip,
            self.target_server_ip, self.client_mapping))
        self.sub_p.start()

    def join(self):
        self.sub_p.join()
        self.in_pipe.close()
        self.out_pipe.close()

    def get_pipe(self):
        return self.out_pipe, self.in_pipe


def tcp_process(pipe, world_size, socket_num, server_ip, client_ip, target_server_ip, client_mapping):
    """

    msg: 'CONNECT' -> server start listening, client start connecting
    msg: 'SEND' -> send data to other party, after receiving 'SEND',  this process need recv target id and data
    msg: 'RECV' -> recv data from other party, after receiving 'RECV', this process need recv target id
    msg: 'CLOSE' -> close all connections

    一个process负责一个server和多个client的连接
    client负责发送数据，server负责接收数据
    每个server和client都有多个socket，每个socket负责一个连接
    server_pipes_out, server_pipes_in: process和server的pipe列表
    client_pipes_out, client_pipes_in: process和client的pipe字典
    server_threads: server的所有线程列表
    client_threads: client的所有线程字典

    client_threads: {target: [thread1, thread2, ...]}

    :param pipe:
    :param world_size:
    :param socket_num:
    :param server_ip: 自己的server ip, 格式：(ip, port)
    :param client_ip: 自己的client ip, 格式：{0: (ip, port), 1: (ip, port), ...}
    :param target_server_ip: 目标server ip, 格式：{0: (ip, port), 1: (ip, port), ...}
    :param client_mapping: 连接自己server的client ip, 格式：{0: (ip, port), 1: (ip, port), ...}
    :return:
    """

    _out_pipe, _in_pipe = pipe
    server_address, server_port = server_ip
    server = TCPServer(server_address, server_port)
    clients = {}
    for target, (client_address, client_port) in client_ip.items():
        client = TCPClient(client_address, client_port)
        clients[target] = client

    server_pipes_out = []
    server_pipes_in = []

    client_pipes_out = {}
    client_pipes_in = {}

    server_threads = []
    client_threads = {}

    while True:
        try:
            msg = _in_pipe.recv()
            if msg == 'CONNECT':
                server.socket_mapping.append({})
                for i in range(socket_num):
                    start_thread(server, world_size, i, server_pipes_out, server_pipes_in, server_threads)
                    server_pipes_in[i].send('CONNECT')
                    time.sleep(3)  # 等待server完全启动 TODO: 端口爆炸问题仍然存在

                    for target, (client_address, client_port) in target_server_ip.items():
                        client = clients.get(target)
                        if client_pipes_out.get(target) is None:
                            client_pipes_out[target] = []
                            client_pipes_in[target] = []
                            client_threads[target] = []
                        client.connect_to_with_retry(client_address, client_port, i)
                        start_thread(client, world_size, i, client_pipes_out[target], client_pipes_in[target],
                                     client_threads[target])
                    time.sleep(1)

                while server.connect_number < (world_size - 1) * socket_num:
                    print("Waiting for all clients to connect...")
                    print(f"Current connect number: {server.connect_number}")
                    time.sleep(1)
                _in_pipe.send('OK')

            elif msg == 'SEND':
                target = _in_pipe.recv()
                data = _in_pipe.recv()
                # data = global_var.get_value("send data")

                # 获取目标client的socket
                tcp = clients.get(target)

                # 将数据分成多份，发送给目标server的多个socket
                data = pickle.dumps(data)

                # 发送数据大小
                message_size = struct.pack("Q", len(data))
                send_binary_data(tcp.client_socket[0], message_size)

                data_size = len(data) // socket_num

                # global_var.set_value('send data b', data)
                # 发送数据
                for i in range(socket_num):
                    # 发送SEND指令
                    client_pipes_in[target][i].send('SEND')
                    client_pipes_in[target][i].send(data[i * data_size: (i + 1) * data_size] if i != socket_num - 1 else
                                                    data[i * data_size:])
                    # client_pipes_in[target][i].send(i * data_size)
                    # client_pipes_in[target][i].send(data_size * (i + 1) if i != socket_num - 1 else len(data))

                for i in range(socket_num):
                    client_pipes_in[target][i].recv()

                # _in_pipe.send('OK')

            elif msg == 'RECV':
                target = _in_pipe.recv()
                target_address = client_mapping.get(target)

                target_socket = server.socket_mapping[0].get(target_address)
                size_data = receive_binary_data(target_socket, struct.calcsize("Q"))
                pure_data_size = struct.unpack("Q", size_data)[0]

                data_size = pure_data_size // socket_num
                last_data_size = pure_data_size - data_size * (socket_num - 1)

                address, port = target_address
                msg_dict = {}
                # global_var.set_value('recv data b', bytearray(pure_data_size))
                for i in range(socket_num):
                    target_socket = server.socket_mapping[i].get((address, port + i))
                    server_pipes_in[i].send('RECV')
                    server_pipes_in[i].send(target_socket)
                    server_pipes_in[i].send(data_size if i != socket_num - 1 else last_data_size)
                    # msg_dict[i] = server_pipes_in[i].recv()

                for i in range(socket_num):
                    msg_dict[i] = server_pipes_in[i].recv()
                # data_placeholder = global_var.get_value('recv data b')
                data_placeholder = b''.join([msg_dict[i] for i in range(socket_num)])

                frame = pickle.loads(data_placeholder)
                # global_var.set_value('recv data', frame)
                _in_pipe.send(frame)

            elif msg == 'CLOSE':
                for i in range(socket_num):
                    server_pipes_in[i].send('CLOSE')
                    server_threads[i].join()
                    server_pipes_in[i].close()
                    server_pipes_out[i].close()
                    for target, client in clients.items():
                        client_pipes_in[target][i].send('CLOSE')
                        client_threads[target][i].join()
                        client_pipes_in[target][i].close()
                        client_pipes_out[target][i].close()
                break
        except EOFError:
            break


def tcp_thread(pipe, tcp, world_size, idx):
    _out_pipe, _in_pipe = pipe

    while True:
        try:
            msg = _out_pipe.recv()
            if msg == 'CONNECT':
                assert isinstance(tcp, TCPServer)
                tcp.start_listening(idx, world_size)
            elif msg == 'SEND':
                assert isinstance(tcp, TCPClient)
                # start = _out_pipe.recv()
                # end = _out_pipe.recv()
                # data = global_var.get_value('send data b')[start:end]
                data = _out_pipe.recv()
                send_data(tcp.client_socket[idx], data)
                _out_pipe.send('OK')
            elif msg == 'RECV':
                assert isinstance(tcp, TCPServer)
                target_socket = _out_pipe.recv()
                data_size = _out_pipe.recv()
                data = receive_data(target_socket, data_size)
                _out_pipe.send(data)
                # data = global_var.get_value('recv data b')
                # if idx != tcp.socket_num - 1:
                #     data[idx * data_size: (idx + 1) * data_size] = receive_data(target_socket, data_size)
                # else:
                #     data[-data_size:] = receive_data(target_socket, data_size)
                # _out_pipe.send('OK')
            elif msg == 'CLOSE':
                break
        except EOFError:
            break


def start_thread(tcp, world_size, idx, outs, ins, threads):
    pipe = Pipe(True)
    out, in_ = pipe
    thread = Thread(target=tcp_thread, args=(pipe, tcp, world_size, idx))
    thread.start()

    outs.append(out)
    ins.append(in_)
    threads.append(thread)
