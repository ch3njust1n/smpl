'''
    Justin Chen
    7.5.17

    Module for handling asynchronous gradient aggregation and sharing

    Boston University
    Hariri Institute for Computing and 
    Computational Sciences & Engineering
'''

import sys
sys.path.insert(0, 'data')

from parameter_channel import ParameterChannel
from multiprocessing import Process, Lock, Manager
from random import random, getrandbits, shuffle, randint
from time import sleep, time
from datetime import datetime
from itertools import combinations
from model.network import NeuralNetwork
import parameter_tools as pt
import os, torch, json, redis, socket, logging, utils, test, train, session, data


class ParameterServer(object):
    def __init__(self, args):
        logging.basicConfig(filename='gradient.log', level=logging.DEBUG)
        self.host     = args['address']['host']
        self.port     = args['address']['port']
        self.clique   = args['clique']-1
        self.cuda     = args['cuda']
        self.data     = args['data']
        self.epochs   = args['epochs']
        self.id       = args['id']
        self.log_freq = args['log_freq']
        self.lr       = args['lr']
        self.max      = args['max']
        self.name     = args['name']
        self.parallel = args['local_parallel']
        self.party    = args['party']
        self.save     = args['save']
        self.scale    = args['scale']
        self.seed     = args['seed']
        self.session  = args['session']
        self.strategy = args['strategy']
        self.ep_count = 0

        # CUDA/GPU settings
        if args.cuda and torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        else:
            torch.manual_seed(args.seed)

        # dictionary for managing gradients exchanged during sessions across processes
        self.manager = Manager()
        self.grad_queue = self.manager.dict()

        # Save all state in Redis
        self.cache = redis.StrictRedis(host='localhost', port=6379, db=0)
        if args['flush']:
            self.flush()

        # Setup parameter cache
        # Network() was named generically intentionally so that users can plug-and-play
        # Track best set of parameters. Equivalent of "global" params in central server model.
        # Stash this server's info
        self.cache.set('best', json.dumps({"accuracy": 0.0, "parameters": [x.data.tolist() for x in NeuralNetwork().parameters()]}))
        self.cache.set('server', json.dumps({"clique": self.clique, "host": self.host, "port": self.port}))

        # Load dataset
        self.dataset = data.SMPLData(self.cuda)

        # Establish ports for receiving API calls
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(1)

        # Setup TCP connections to all peers
        self.roster = utils.load_json(os.path.join(os.getcwd(), 'distributed/config/', self.party))
        self.peers = self.roster[1:]
        self.me = self.roster[0]
        self.pc = ParameterChannel(self.peers)

        # Init training
        time.sleep(random())
        self.async_train()


    '''
    Check if the party configuration file is correct
    '''
    def check_party_config(self):
        # Check that all aliases are unique
        pass


    '''
    Listen for incoming messages from party members
    '''
    def listen(self):
        logging.info('listening on port %d' % self.port)
        try:
            while True:
                if self.ep_count == self.epochs or len(self.pc.connections) == 0:
                    self.pc.teardown()
                    break

                conn, addr = self.sock.accept()
                logging.info(datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S'))
                logging.info('from: '+str(addr))
                p = Process(target=self.receive, args=(conn, addr))
                p.start()

        except KeyboardInterrupt:
            print '\nexiting...'
            sys.exit(0)


    '''
    Input: conn (tuple), add (tuple)
    '''
    def receive(self, conn, addr):
        try:
            resp = {}

            packet = conn.recv(4096)

            # if peer closes connection, then remove that peer from PC connections
            if len(packet) == 0:
                self.pc.remove(addr[0]+':'+addr[1])

            msg = packet.split('::')

            if len(msg) < 2:
                logging.fatal('gs.receive(): empty message')
                conn.sendall('invalid protocol')
                return

            expected = int(msg[0])
            data = msg[1]

            while len(data) < expected:
                # TODO change this to array join, string concat will get expensive if packets are large
                data += conn.recv(min(expected - len(data), 4096))

            resp = self.route({"addr": addr, "length": expected, "content": json.loads(data)})

            if resp == 'invalid':
                logging.info('invalid message: ', data, ' from ', str(addr))

            conn.sendall(self.format_msg(resp))
        except ValueError as e:
            logging.fatal(e)
            conn.sendall('invalid protocol')
        finally:
            logging.info('closing connection')
            conn.close()


    '''
    Input: msg (str)
    Output: str
    '''
    def format_msg(self, msg):
        msg = json.dumps(msg)
        return ''.join([str(len(msg)), '::', str(msg)])


    '''
    Map the incoming API call to the correct function

    Input:  msg (str)
    Output: response (str)
    '''
    def route(self, msg):
        # sender_ip = msg['addr'][0]
        # sender_port = msg['addr'][1]

        content = msg['content']
        api = content['api']
        args = content['args'] if 'args' in content else None

        if api == 'establish_session':
            return self.establish_session(*args)
        elif api == 'synchronize_parameters':
            return self.synchronize_parameters(*args)
        elif api == 'get_parameters':
            return self.get_parameters(*args)
        elif api == 'share_grad':
            return self.share_grad(*args)
        elif api == 'ring_allreduce':
            return self.ring_allreduce(*args)
        else:
            return 'invalid'


    '''
    Wrapper function for train()
    '''
    def async_train(self):
        Process(target=self.train_hyperedge).start()


    '''
    Async establish clique and synchronize parameters
    '''
    def train_hyperedge(self):
        connected = False
        sess_id = 0

        # establish clique
        while not connected:
            connected, sess_id = self.init_session()

        # setup session in gradient queue
        self.grad_queue[sess_id] = {"peers":[], "gradients":[]}

        # each session should create its own model
        nn = NeuralNetwork()

        # pull current best parameters from parameter server
        best = json.loads(self.cache.get('best'))
        nn.update_parameters(best['parameters'])

        self.train(sess_id, nn)

        # if using divergent exploration, then need to update the network parameters
        if self.parallel == 'dex':
            nn.update_parameters(self.cache.get(sess_id)['parameters'])

        ### CHCECKING ###
        avg_grad = self.share_gradient(sess_id)

        # validate model and validate
        nn.update_parameters(self.nn.get_parameters(), avg_grad)

        ### CHCECKING ###
        # update best model on parameter server
        self.update_model(sess_id)

        # clean up parameter cache and gradient queue
        self.cache.delete(sess_id)
        del self.grad_queue[sess_id]

        # increment total successful training epoches
        with Lock():
            self.ep_count += 1


    '''
    Internal API
    Average gradients, distribute to other peers, and update gradients in model

    Input:  sess_id (str)
            model   (NeuralNet)
    '''
    def average_gradients(self, sess_id, model):
        gradients = model.get_sparse_gradients(tolist=True)
        sess = json.loads(self.cache.get(sess_id))
        master = sess['master']

        if self.me == master:

            # wait until all peers have shared the gradients
            grad_queue = self.grad_queue[sess_id]
            while len(grad_queue['peers']) < len(sess['party']):
                pass

            # update model gradients
            model.add_batched_coordinates(grad_queue['gradients'])
            avg_grads = [g/(self.clique+1) for g in model.get_gradients()]
            
            # send to all peers
            # MISSING CODE
        
        else:
            send_to = master
            # if you're not master, send your gradients to master and wait for master to send you the average
            avg_grads = self.pc.send({"api": "share_grad", "args": [sess_id, self.me, gradients]}, 
                                      send_to['host'], send_to['port'])

            # update your model gradients with the average
            model.add_batched_coordinates(avg_grads)


    '''
    Public API 
    Averages gradients across party.
    '''
    def share_grad(self, sess_id, peer, gradients):
        sess = json.loads(self.cache.get(sess_id))
        grad_queue = self.grad_queue[sess_id]
        grad_queue['peers'].append(peer['alias'])
        grad_queue['gradients'].extend(gradients)


    def train(self, sess_id, nn):
        '''
        - Hyper-parallelize with Hogwild!
        - Pass sess_id to Train so it can retrieve the session object from redis
        - Can define a more specific training schedule by passing self.epochs to Train 
          default is 5 epochs of local training before synchronizing gradients
        '''
        if self.parallel == 'hogwild':
            nn.share_memory()

        processes = []

        for c in range(mp.cpu_count()):
            lr = 10**(-c - self.lr) if self.parallel == 'dex' else self.lr
            t = Train(sess_id, nn, self.epochs, lr, self.dataset, self.cache, self.parallel)
            p = mp.Process(target=t.train, args=(average_gradients))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()


    '''
    Get parameters from a party and update parameters

    Input:  sess_id (str), session (str), peers (str)
    Output: ok (bool)
    '''
    def synchronize_parameters(self, sess_id, best, peers):
    
        ok = False

        # If my parameters do not have the best validation accuracy
        best_params = None
        if best['host'] != self.me['host']:

            # Always only wanna synchronize with best set of parameters
            ok, resp = self.pc.send({"api": "get_parameters", "args":["best"]}, best["host"], best["port"])

            if len(resp) > 0:
                best_params = resp[0]
                self.cache.set(sess_id, json.dumps({"parameters": resp[0], "accuracy": resp[1], "peers": peers}))
        else:
            best = json.loads(self.cache.get('best'))
            best['party'] = peers
            best_params = best['parameters']
            self.cache.set(sess_id, json.dumps(best))

        # Start locally training
        nn = NeuralNetwork()
        nn.update_parameters(best_params)
        Process(target=train, args=(sess_id, nn,)).start()

        return ok


    '''
    Only called from main process in train.train() for establishing a training session
    Output:
    '''
    def init_session(self):
        ok = False
        sess_id = ''.join(['sess', str(getrandbits(randint(1,256)))])
        sess = {"id": sess_id, "me": self.me}
        resp = []

        peers = [x for x in self.establish_clique(sess, self.pc) if len(x) > 0]

        # if can't connect with other peers, respond indicating failure
        if len(peers) < 1:
            # remove dead session from cache
            self.cache.delete(sess_id)
            return False, sess_id

        # sort by accuracy in descending order and cache as current session
        peers = sorted(peers, key=lambda x: x['accuracy'], reverse=True)
        best = peers[0]

        # Synchronize parameters of model with best validation accuracy
        resp = []
        for send_to in peers:
            Process(target=self.pc.send, 
                    args=({"api": "synchronize_parameters", "args": [sess_id, best, peers]}, 
                           send_to['host'], send_to['port'],)).start()
            
        # request parameters from member with highest accuracy
        model = {}
        if best['alias'] != sess.me['alias']:
            ok, model = self.pc.send({"api": "get_parameters", "args": ['best']}, 
                                        best['host'], best['port'])
        else:
            model = json.loads(self.cache.get('best'))

        # save parameters so can calculate difference (gradient) after training
        self.cache.set(sess_id, json.dumps({"parameters": model['parameters'], "accuracy": model['accuracy'], 
                                            "master": self.me, "party": peers, "pid": 0, "val": [0.0], "losses": []}))

        return ok, sess_id


    '''
    Update the accuracy and parameters of a session

    Input: sess_id (str), parameters (str), accuracy (float)
    '''
    def update_model(self, sess_id):
        model = json.loads(self.cache.get(sess_id))

        if sess_id == 'best':
            if model['accuracy'] < accuracy:
                model['parameters'] = parameters
                model['accuracy'] = accuracy
                self.cache.set(sess_id, json.dumps(model))
        else:
            model['parameters'] = parameters
            model['accuracy'] = accuracy
            self.cache.set(sess_id, json.dumps(model))


    '''
    Get all active sessions

    Output: list of tuples of sessions
    '''
    def active_sessions(self):
        active_ids = [k for k in self.cache.scan_iter('sess*')]
        sessions = self.cache.mget(active_ids)
        # [(<sess_id>, {"parameters": model[0], "accuracy": model[1], "party": peers}), ...]
        # peers = [{alias, host, port, accuracy}, ...]
        return zip(active_ids, sessions)


    '''
    Check that the given session does not overlap with the currently running sessions

    Input:  peers (list) List of dicts. See /distributed/config/party.json.
    Output: clique (list) List of dics. 
    '''
    def get_unique_clique(self, peers):
        possible_cliques = list(combinations(peers, self.clique))
        shuffle(possible_cliques)
        # possible_cliques: [({u"alias": u"moc", u"host": u"128.31.25.26", u"port": 9888}, 
        #                     {u"alias": u"moc-2", u"host": u"128.31.24.191", u"port": 9888})]
        clique = []
        active = self.active_sessions()
        sess_hash = [hash(a) for a in active]

        if len(active) == 0:
            return possible_cliques.pop(0)

        # Check to ensure that overlapping cliques are not formed
        while possible_cliques:
            clique = possible_cliques.pop(0)

            if hash(str(clique)) not in sess_hash:
                return list(clique)

        return clique


    '''
    Establish a training clique
    Input: sess (Session)
    Output: {alias, host, port, accuracy} to init_session()
    '''
    def establish_clique(self, sess):
        # find a unique clique
        unique = self.get_unique_clique(self.peers)
        peers = []

        # Note: Parallelize this!!!
        for send_to in unique:
            ok, resp = self.pc.send({"api": "establish_session", "args": [sess.id, sess.me]}, 
                                        send_to['host'], send_to['port'])

            if ok and len(resp) > 0:
                peers.append(resp)

            if len(peers) >= self.clique:
                break

        return peers[:self.clique]


    '''
    Reply to request to establish a session
    Input: sess_id (str), master (dict)
    Output: {alias, host, port, accuracy}
    '''
    def establish_session(self, sess_id, master):
        record = json.loads(self.cache.get('best'))
        
        me = dict(self.me)
        me['accuracy'] = record['accuracy']
        self.cache.set(sess_id, json.dumps({"id": sess_id, "master": master, "peers": []}))
        return me


    '''
    API for getting this member's parameters
    Inputs: sess_id (str)
    Output: parameters (list) Nested list of lists
            accuracy (float)
    '''
    def get_parameters(self, sess_id, sparsity=0):
        model = json.loads(self.cache.get(sess_id))
        
        if model == None:
            return [], -1

        # CALL DEEP GRADIENT COMPRESSION HERE
        if sparsity > 0:
            model = json.loads(model)
            return pt.largest_k(model['parameters'], sparsity), model['accuracy']
        else:
            return model['parameters'], model['accuracy']


    '''
    Input: signal, frame
    '''
    def force_stop(self, signal, frame):
        self.stop()


    '''
    Deletes everything in the redis cache
    '''
    def flush(self):
        self.cache.flushall()


    '''
    Stops the parameter server
    '''
    def stop(self):
        try:
            self.sock.close()
            logging.info('closing sockets')
            sys.exit(0)
        except Exception as e:
            logging.Fatal('Could not close ParameterServer socket')
            return False
        return True