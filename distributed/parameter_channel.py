'''
	Justin Chen

	7.9.17

	Module of handling communication between the training loop and the ParameterServer

	Boston University
	Hariri Institute for Computing and 
    Computational Sciences & Engineering
'''

import socket, logging, json
from multiprocessing import Process, Manager


# Channel for sending to and receiving gradients from MPC server
class ParameterChannel(object):
    def __init__(self, peers):
        logging.basicConfig(filename='gradient.log', level=logging.DEBUG)
        self.peers = peers
        self.connections = {}
        self.setup()


    '''
    Initial setup for connecting to all peers
    '''
    def setup(self):
        for p in self.peers:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.settimeout(0.1)
                sock.connect((p['host'], p['port']))
                address = p['host']+':'+p['port']
                self.connections[address] = sock
            except socket.error as sock_err:
                if (sock_err.errno == socket.errno.ECONNREFUSED):
                    logging.info(sock_err)


    '''
    Format API messages

    Input:  msg (dict) API function call and parameters
    Output: (string) response
    '''
    def format(self, msg):
        msg = json.dumps(msg)
        return ''.join([str(len(msg)), '::', str(msg)])


    '''
    Use to communicate with peer via TCP

    Input:  peer (int) Integer indicating peer
            msg (dict) API function call and parameters
    Output: ok (bool)
            content (dict)
    '''
    def send(self, peer, msg):
        ok = False

        try:
            resp = ''
            # Send filename containing gradient to Go server
            self.sock.sendall(self.format(msg) + '\n')
            
            # Look for the response
            resp = self.sock.recv(4096).split('::')
            expected = int(resp[0])
            content = resp[1]
            received = len(content)
            remaining = expected - received

            while len(content) < expected:
                content += self.sock.recv(min(expected - len(content), 4096))
                received = len(content)

            # Received entire message
            ok = received == expected

        return ok, json.loads(content)


    '''
    Remove peer's socket
    '''
    def remove(self, peer):
        sock = self.connections[peer]
        sock.shutdown(socket.SHUT_RDWR)
        sock.close()
        del self.connections[peer]


    '''
    Teardown connections to all peers
    '''
    def teardown(self):
        for sock in self.connections:
            self.remove(peer)

        logging.info('closed all connections')
