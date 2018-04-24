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
        self.log = logger
        self.peers = peers
        self.connections = {}
        self.status = 1
        # self.setup()


    '''
    Connect to a single peer

    Input: peer (dict) Dictionary contains address information
    '''
    def connect(self, peer):

        address = '{}:{}'.format(peer['host'], peer['port'])
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # sock.settimeout(0.1)
        attempts = 0

        while address not in self.connections:

            attempts += 1
            try:
                self.log.debug('host:{}, port:{}'.format(peer['host'], peer['port']))
                sock.connect((peer['host'], peer['port']))
                self.connections[address] = sock
            except socket.error as sock_err:
                sleep(1)
                self.log.error('{}, addr: {}'.format(str(sock_err), address))
            except KeyError as key_err:
                self.log.error(str(key_err))

        self.log.info('success {} connected'.format(address))


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
    Send message to all peers in parallel. This function will not indicate if all
    messages were sent.

    Input:  msg   (string) Message to be sent
            peers (list)   List of dictionaries with peer information
    '''
    def sendall(self, peers, api, arguments):
        for send_to in peers:
            Thread(target=self.send, args=(send_to['host'], send_to['port'], 
                                          {"api": api, "args": arguments},)).start()


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
            sock = self.connections[addr]
            msg = self.format(msg) + '\n'
            sock.sendall(msg)
            
            # Look for the response
            resp = sock.recv(4096).split('::')
            self.log.debug('resp:{} type:{}'.format(resp, type(resp)))
            if len(resp[0]) > 0:
                expected = int(resp[0])
                content = resp[1]
                self.log.debug('expected: {} content:{}'.format(expected, content))
                received = len(content)
                remaining = expected - received

                while len(content) < expected:
                    content += sock.recv(min(expected - len(content), 4096))
                    received = len(content)

                # Received entire message
                ok = received == expected
                content = ujson.loads(content)
            else:
                self.log.error('empty reply: {} for api:{}'.format(resp, msg))
                content = ''

        except Exception as e:
            self.log.exception(str(e))

        return ok, content


    '''
    Remove a particular peer from the active connections list
    '''
    def remove(self, peer):

        try:
            sock = self.connections[peer]
            sock.close()
            del self.connections[peer]
            self.log.info('closed socket {}'.format(peer))
        except KeyError as e:
            self.log.error('{}\n{}'.format(e, self.connections))


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

        self.log.info('closed all connections')
