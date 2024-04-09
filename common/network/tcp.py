import socket
import pickle
import struct
import numpy as np
import torch

from config.base_configs import DEVICE
from config.network_config import TCP_BUFFER_SIZE


class TCPServer(object):
    def __init__(self, addr, port):
        addr='0.0.0.0'
        self.address = (addr, port)
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(self.address)
        self.server_socket.listen(1)
        self.client_socket = None
        self.buffer = None
        self.is_connected = False
        self.payload_size = struct.calcsize("L")
        self.data = b''

    def run(self):
        print("TCPServer waiting for connection ......")
        self.client_socket, client_addresss = self.server_socket.accept()
        self.is_connected = True
        print("TCPServer successfully connected by :%s" % str(client_addresss))

    def send_msg(self, msg):
        self.client_socket.send(msg)

    def send_value(self, v):
        package = np.array([v])
        self.send_np_array(package)

    def send_np_array(self, array):
        data = pickle.dumps(array)
        message_size = struct.pack("L", len(data))
        self.client_socket.sendall(message_size + data)

    def send_torch_array(self, array):
        array = array.cpu().detach().numpy()
        self.send_np_array(array)

    def receive_msg(self):
        recvstr = self.client_socket.recv(TCP_BUFFER_SIZE)
        return recvstr

    def receive_value(self):
        return self.receive_np_array()[0]

    def receive_np_array(self):
        while len(self.data) < self.payload_size:
            self.data += self.client_socket.recv(self.payload_size)

        packed_msg_size = self.data[:self.payload_size]
        self.data = self.data[self.payload_size:]
        msg_size = struct.unpack("L", packed_msg_size)[0]

        # Retrieve all data based on message size
        while len(self.data) < msg_size:
            self.data += self.client_socket.recv(msg_size)

        frame_data = self.data[:msg_size]
        self.data = self.data[msg_size:]

        # Extract frame
        frame = pickle.loads(frame_data)
        # frame = frame.astype(int)
        return frame

    def receive_torch_array(self):
        frame = self.receive_np_array()
        frame = torch.from_numpy(frame).to(DEVICE)
        return frame

    def close(self):
        self.client_socket.close()


class TCPClient(object):
    def __init__(self, host, port):
        self.address = (host, port)
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        self.server_socket = None
        self.buffer = None
        self.is_exit = False
        self.is_connected = False
        self.payload_size = struct.calcsize("L")
        self.data = b''

    def run(self):
        self.client_socket.connect(self.address)
        print('successfully connected to server: %s' % str(self.address[0]))
        self.is_connected = True

    def send_msg(self, msg):
        self.client_socket.send(msg)

    def send_value(self, v):
        package = np.array([v])
        self.send_np_array(package)

    def send_np_array(self, array):

        # print(array.size)

        data = pickle.dumps(array)
        message_size = struct.pack("L", len(data))
        self.client_socket.sendall(message_size + data)

    def send_torch_array(self, array):
        array = array.cpu().detach().numpy()
        self.send_np_array(array)

    def receive_msg(self):
        recvstr = self.client_socket.recv(TCP_BUFFER_SIZE)
        return recvstr

    def receive_value(self):
        return self.receive_np_array()[0]

    def receive_np_array(self):
        while len(self.data) < self.payload_size:
            self.data += self.client_socket.recv(self.payload_size)

        packed_msg_size = self.data[:self.payload_size]
        self.data = self.data[self.payload_size:]
        msg_size = struct.unpack("L", packed_msg_size)[0]

        # Retrieve all data based on message size
        while len(self.data) < msg_size:
            self.data += self.client_socket.recv(msg_size)

        frame_data = self.data[:msg_size]
        self.data = self.data[msg_size:]

        # Extract frame
        frame = pickle.loads(frame_data)
        # frame = frame.astype(int)
        return frame

    def receive_torch_array(self):
        frame = self.receive_np_array()
        frame = torch.from_numpy(frame).to(DEVICE)
        return frame

    def close(self):
        self.client_socket.close()


def create_tcp(addr, port, type):
    if type == 'server':
        return TCPServer(addr, port)
    elif type == 'client':
        return TCPClient(addr, port)
    else:
        raise RuntimeError('Please make sure the tcp type is in one of [server，client]')
