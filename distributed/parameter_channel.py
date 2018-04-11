'''
	Justin Chen

	7.9.17

	Module of handling communication between the training loop and the ParameterServer
'''

import socket, ujson
from multiprocessing import Process, Manager
from threading import Thread
from time import sleep

# Channel for sending to and receiving gradients from MPC server
class ParameterChannel(object):

    def __init__(self, peers, logger=None):
        self.logger = logger
        self.peers = peers
        self.connections = {}
        self.status = 1
        self.setup_tries = 10
        self.setup()


    '''
    Connect to a single peer

    Input: peer (dict) Dictionary contains address information
    '''
    def connect(self, peer):

        address = '{}:{}'.format(peer['host'], peer['port'])
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # sock.settimeout(0.1)
        attempts = 0

        self.logger.info('contacting: {}'.format(address))
        while address not in self.connections:

            # if attempts == self.setup_tries: break

            try:
                sock.connect((peer['host'], peer['port']))
                self.connections[address] = sock
            except Exception as sock_err:
                self.logger.info('Error: pc sock_err:{}'.format(sock_err))

                if (sock_err.errno == socket.errno.ECONNREFUSED):
                    sleep(1)
                    self.logger.info('Error: pc.setup() {}, addr: {}'.format(str(sock_err), address))
                    attempts += 1

        self.logger.info('sucess {} connected'.format(address))


    '''
    Initial setup for connecting to all peers
    '''
    def setup(self):
        queue = []
        
        for p in self.peers:
            t = Thread(target=self.connect, args=(p,))
            t.start()
            queue.append(t)

        for t in queue:
            t.join()


    '''
    Format API messages

    Input:  msg (dict) API function call and parameters
    Output: (string) response
    '''
    def format(self, msg):
        return ''.join([str(len(msg)), '::', ujson.dumps(msg)])


    '''
    Use to communicate with peer via TCP

    Input:  host (string) IP address
            port (int) Port number
            msg (dict) API function call and parameters
    Output: ok (bool)
            content (dict)
    '''
    def send(self, host, port, msg):
        ok = False
        content = ''

        try:
            resp = ''
            addr = '{}:{}'.format(host, port)

            self.logger.info('pc.send() addr:{}'.format(addr))

            sock = self.connections[addr]

            msg = self.format(msg) + '\n'
            self.logger.info('ps.send() msg:{}'.format(msg))
            sock.sendall(msg)

            self.logger.info('pc.send() sent!')
            
            # Look for the response
            resp = sock.recv(4096).split('::')

            self.logger.debug('pc.send() resp[0]:{}'.format(resp[0]))
            expected = int(resp[0])
            
            self.logger.info('pc.send() expected:{}'.format(expected))

            content = resp[1]

            received = len(content)
            remaining = expected - received

            self.logger.info('pc.send() looking')

            while len(content) < expected:
                content += sock.recv(min(expected - len(content), 4096))
                received = len(content)

            self.logger.info('pc.send() gotit!')

            # Received entire message
            ok = received == expected

            content = ujson.loads(content)

            self.logger.info('pc.send() ok:{}'.format(ok))

        except Exception as e:
            self.logger.info('Error: pc.send() '+str(e))

        return ok, content


    '''
    Remove a particular peer from the active connections list
    '''
    def remove(self, peer):

        try:
            sock = self.connections[peer]
            sock.close()
            del self.connections[peer]
            self.logger.info('closed socket {}'.format(peer))
        except KeyError as e:
            self.logger.debug('{}\n{}'.format(e, self.connections))


    '''
    Teardown connections to all peers
    '''
    def teardown(self):
        
        for sock in self.connections:
            sock = self.connections[peer]
            sock.shutdown(socket.SHUT_RDWR)
            sock.close()

            if len(self.connections) == 0:
                self.status = 0
                self.teardown()

        self.connections.clear()

        self.logger.info('closed all connections')
