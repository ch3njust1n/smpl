'''
	Justin Chen

	7.9.17

	Module of handling communication between the training loop and the ParameterServer
'''

import socket, ujson
from multiprocessing import Manager
from threading import Thread
from time import sleep
import copy_reg
from multiprocessing.reduction import rebuild_socket, reduce_socket
copy_reg.pickle(socket.socket, reduce_socket, rebuild_socket)


# Channel for sending to and receiving gradients from MPC server
class ParameterChannel(object):

    def __init__(self, peers, logger=None):
        self.log = logger
        self.peers = peers
        self.connections = Manager().dict()
        self.status = 1


    '''
    Connect to a single peer

    Input: peer (dict) Dictionary contains address information
    '''
    def connect(self, peer):

        address = '{}:{}'.format(peer['host'], peer['port'])
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.log.debug('addr: {}, sock: {}, connections: {}'.format(address, sock, self.connections))

        while address not in self.connections:

            try:
                self.log.debug('host:{}, port:{}'.format(peer['host'], peer['port']))
                sock.connect((peer['host'], peer['port']))
                # self.connections[address] = sock
                tmp = self.connections
                tmp[address] = sock
                self.connections = tmp
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

        self.log.debug('connections: {}'.format(self.connections))


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

    Input: peers (list)   List of dictionaries with peer information
           api   (string) API to call
           args  (list)   API arguments
           sync  (bool)   If True, will send synchronously and return all responses
    '''
    def sendall(self, peers, api, arguments, sync=False):
        if sync:
            return [self.send(send_to['host'], send_to['port'], {"api": api, "args": arguments}) 
                    for send_to in peers]
        else:
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
            
            self.log.debug('wherethefuckami1')
            if addr not in self.connections:
                self.log.error('addr: {} not in connections: {}'.format(addr, self.connections))
                return False, ''

            self.log.debug('wherethefuckami2')
            sock = self.connections[addr]
            msg = self.__format_msg(msg)
            sock.sendall(msg)
            
            self.log.debug('wherethefuckami3 {}'.format(host))
            # Look for the response
            resp = sock.recv(4096).split('::')

            self.log.debug('wherethefuckami4')
            if 'invalid' in resp:
                self.log.debug('addr: {}, msg: {}'.format(addr, msg))
            
            # Check that a length is given
            if len(resp[0]) > 0:
                expected = 0

                self.log.debug('wherethefuckami5')
                try:
                    expected = int(resp[0])
                except ValueError as e:
                    self.log.error('unspecified message length: {}'.format(resp))
                    return False, ''

                self.log.debug('wherethefuckami6')

                content = resp[1]
                received = len(content)
                remaining = expected - received

                self.log.debug('wherethefuckami7')

                if remaining < 0:
                    self.log.debug('wherethefuckami8')
                    self.log.error('received more than expected')
                    raise Exception('received more than expected')

                self.log.debug('wherethefuckami9')
                while len(content) < expected:
                    packet = sock.recv(expected - len(content))
                    if not packet:
                        self.log.error('packet error: {}'.format(packet))
                        return False, ''
                    content += packet

                self.log.debug('wherethefuckami10')
                # Received entire message
                received = len(content)
                ok = received == expected
                self.log.debug('received ({}), expected ({})'.format(received, expected))
                
                self.log.debug('wherethefuckami11')
                if not ok:
                    msg = 'Did not receive entire response. received:{} expected:{}'.format(received, expected)
                    self.log.error(msg)
                    raise Exception(msg)
                    return False, ''
                self.log.debug('wherethefuckami12')
                content = ujson.loads(content)
                self.log.debug('wherethefuckami13')
            else:
                self.log.debug('wherethefuckami14')
                self.log.error('empty reply: {} for api:{}'.format(resp, msg))
                content = ''
        except Exception as e:
            self.log.debug('wherethefuckami15')
            self.log.exception(str(e))

        self.log.debug('wherethefuckami16')
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
