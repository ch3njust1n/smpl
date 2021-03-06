'''
    Justin Chen

    7.9.17

    Module of handling communication between the training loop and the ParameterServer
'''

import socket, json, select
from multiprocessing import Manager
from threading import Thread
from random import random
from time import sleep
import copy_reg
from multiprocessing.reduction import rebuild_socket, reduce_socket
copy_reg.pickle(socket.socket, reduce_socket, rebuild_socket)


# Channel for sending to and receiving gradients from MPC server
class ParameterChannel(object):

    def __init__(self, peers, logger=None):
        self.log = logger
        self.peers = peers if type(peers) == list else [peers]
        self.connections = Manager().dict()
        self.status = 1


    '''
    Format peer's host and port information

    Input:  peer (dict, tuple, or list) Object containing peer information
    Output: addr (string) Formatted address key
    '''
    def addr_key(self, peer):
        if isinstance(peer, dict):
            return (peer['host'], peer['port']), '{}:{}'.format(peer['host'], peer['port'])
        elif isinstance(peer, tuple) or isinstance(peer, list):
            return tuple(peer), '{}:{}'.format(peer[0], peer[1])
        else:
            raise Exception('invalid type')


    '''
    Connect to a single peer

    Input: peer (dict) Dictionary contains address information
    '''
    def connect(self, peer):
        addr_tup, addr_key = self.addr_key(peer)

        if addr_key in self.connections:
            del self.connections[addr_key]

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        while addr_key not in self.connections:

            try:
                sock.connect(addr_tup)
                self.connections[addr_key] = sock
            except socket.error as err:
                sleep(1)
                self.log.error('{}, addr: {}'.format(str(err), addr_key))

                '''
                ENETDOWN        Network is down
                ENETUNREACH     Network is unreachable
                ECONNABORTED    Connection reset by peer
                EHOSTUNREACH    No route to host
                
                Python TCP socket error message:
                https://docs.python.org/2/library/errno.html#module-errno
                '''
                if err == socket.errno.ENETDOWN or err == socket.errno.ENETUNREACH or err == socket.errno.ECONNABORTED or err == socket.errno.EHOSTUNREACH:
                    break

            except KeyError as key_err:
                self.log.error(str(key_err))

        self.log.info('success {} connected'.format(addr_key))
        return addr_key in self.connections



    '''
    Wrapper for reconnecting so that can log this API call

    Input: peer (tuple) Tuple containing a peer's host and port information
    '''
    def reconnect(self, peer):
        self.log.info('reconnecting to {}'.format(peer))
        self.connect(peer)


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
    def format_msg(self, msg):
        msg = json.dumps(msg)
        size = 0
        try:
            size = len(msg)
        except TypeError as e:
            msg = str(msg)
            size = len(msg)
        return '{}::{}'.format(size, msg)


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
        api = msg['api']

        try:
            resp = ''
            addr = '{}:{}'.format(host, port)
            
            if addr not in self.connections:
                self.log.error('addr: {} not in connections: {}'.format(addr, self.connections))
                return False, ''

            sock = self.connections[addr]
            msg = self.format_msg(msg)

            try:
                sock.sendall(msg)
                resp = sock.recv(4096).split('::')
            except socket.error, e:
                sock.close()

                '''
                ECOMM       Communication error on send
                ECONNRESET  Connection reset by peer
                EPIPE       Broken pipe

                Python TCP socket error message:
                https://docs.python.org/2/library/errno.html#module-errno
                '''
                if e.errno == socket.errno.ECONNRESET or e.errno == socket.errno.EPIPE or socket.errno.ECOMM:
                    self.reconnect((host, port))
                    return False, ''
                else:
                    self.log.exception(e)
                    del self.connections[addr]
                    return False, ''
            
            # Check that a length is given
            if len(resp) > 0 and len(resp[0]) > 0:
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
                        self.log.error('packet error: {}'.format(packet))
                        return False, ''
                    content += packet

                # Received entire message
                received = len(content)
                ok = received == expected
                
                if not ok:
                    msg = 'Did not receive entire response. received:{} expected:{}'.format(received, expected)
                    raise Exception(msg)
                    return False, ''

                content = json.loads(content)

                if content == None:
                    return False, ''

            else:
                self.log.error('empty reply from {} for api: {}'.format(addr, api))
                content = ''
        except Exception as e:
            self.log.exception('api: {} exception: {}'.format(api, str(e)))

        return ok, content


    '''
    Wraper for send() to persistently, hence psend, retry until message is successfully sent.
    This should be called in a Thread or Process so that it does not tie up the initiating process.

    Input:  host    (string) IP address
            port    (int)    Port number
            msg     (dict)   API function call and parameters
    Output: ok      (bool)
            content (dict)
    '''
    def psend(self, host, port, msg):
        resp = None
        ok = False

        self.log.info('calling api: {}'.format(msg['api']))

        while resp == None:
            ok, resp = self.send(host, port, msg)

        self.log.info('called {}'.format(msg['api']))

        return ok, resp


    '''
    Remove a particular peer from the active connections list

    Input: peer ()
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


    '''
    Return all active connections
    '''
    def __str__(self):
        return str(self.connections)


    '''
    Return total number of existing connections
    '''
    def __len__(self):
        return len(self.connections)