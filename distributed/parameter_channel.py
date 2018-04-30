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
    def __format_msg(self, msg):
        msg = ujson.dumps(msg)
        return '{}::{}'.format(len(msg), msg)


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
            
            if addr not in self.connections:
                return False, ''

            sock = self.connections[addr]
            msg = self.__format_msg(msg)
            sock.sendall(msg)
            
            # Look for the response
            resp = sock.recv(4096).split('::')
            
            # Check that a length is given
            if len(resp[0]) > 0:
                expected = 0

                try:
                    expected = int(resp[0])
                except ValueError as e:
                    self.log.error('unspecified message length: {}'.format(resp))
                    return False, ''

                content = resp[1]
                received = len(content)
                remaining = expected - received

                if remaining < 0:
                    raise Exception('received more than expected')

                while len(content) < expected:
                    packet = sock.recv(expected - len(content))
                    if not packet:
                        return False, ''
                    content += packet

                # Received entire message
                received = len(content)
                ok = received == expected
                self.log.debug('received ({}), expected ({})'.format(received, expected))
                
                if not ok:
                    raise Exception('Did not receive entire response. received:{} expected:{}'.format(received, expected))
                    return False, ''
                content = ujson.loads(content)
                # self.log.debug('addr: {}, api: {}'.format(addr, msg))
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

        if peer in self.connections:
            sock = self.connections[peer]
            sock.close()
            del self.connections[peer]
            self.log.info('closed socket {}'.format(peer))
        else:
            self.log.info('Could not remove peer: {}'.format(peer))


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
