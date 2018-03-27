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
from multiprocessing import Process, Lock, Manager, cpu_count
from threading import Thread
from random import random, getrandbits, shuffle, randint
from time import sleep, time
from train import Train
from datetime import datetime
from itertools import combinations
from model.network import NeuralNetwork, TestNeuron
import parameter_tools as pt
import os, torch, json, redis, socket, logging, utils, test, train, data


class ParameterServer(object):
    def __init__(self, args):
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        handler = logging.FileHandler(os.path.join(os.getcwd(),'logs/gradient.log'))
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        self.host           = args.host
        self.port           = args.port
        self.workers        = (cpu_count()-args.clique)/(args.clique)
        self.clique         = args.clique-1
        self.cuda           = args.cuda
        self.data           = args.data
        self.dev            = args.dev
        self.epochs         = args.epochs
        self.eth            = args.eth
        self.epsilon        = args.epsilon
        self.log_freq       = args.log_freq
        self.max            = args.max
        self.name           = args.name
        self.parallel       = args.local_parallel
        self.party          = args.party
        self.save           = args.save
        self.seed           = args.seed
        self.sparsity       = args.sparsity
        self.strategy       = args.strategy
        self.train_path     = os.path.join(args.data, 'train')
        self.train_rank     = args.train_rank
        self.val_path       = os.path.join(args.data, 'val')
        self.val_rank       = args.val_rank
        self.ep_count       = 0
        self.lock           = Lock()

        self.clear_port()

        if self.dev:
            # CUDA/GPU settings
            if args.cuda and torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
            else:
                torch.manual_seed(args.seed)

        # dictionary for managing gradients exchanged during sessions across processes
        self.manager = Manager()
        self.grad_queue = self.manager.dict()

        # Save all state in Redis
        self.cache = redis.StrictRedis(host='localhost', port=6379, db=0)
        if args.flush:
            self.flush()

        # Setup parameter cache
        # Network() was named generically intentionally so that users can plug-and-play
        # Track best set of parameters. Equivalent of "global" params in central server model.
        # Stash this server's info
        self.cache.set('best', json.dumps({"accuracy": 0.0, "val_size": 0, "train_size": 0, "rank": 100,
                                           "parameters": [x.data.tolist() for x in TestNeuron().parameters()]}))
        self.cache.set('server', json.dumps({"clique": self.clique, "host": self.host, "port": self.port}))

        # Establish ports for receiving API calls
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(1)
        Thread(target=self.listen).start()

        # Load party config
        self.roster = utils.load_json(os.path.join(os.getcwd(), 'distributed/config/', self.party))
        
        if self.clique == len(self.roster):
            raise Exception('SMPL only supports simple hypergraphs: clique size < numer of peers (k  < |V|)')

        utils.check_party(self.roster)

        # Setup TCP connections to all peers
        self.peers = self.roster[1:]

        if len(self.peers) < self.clique:
            raise Exception('Error: clique size must be less than or equal to the number of peers')

        self.me = utils.get_me(self.roster, eth=self.eth)
        self.pc = ParameterChannel(self.peers, logger=self.logger)

        # Load dataset
        self.dataset = data.SMPLData(self.data, cuda=self.cuda, shares=len(self.roster), index=self.me['id'])

        # Init training
        sleep(random())
        self.async_train()

        # Init local training
        sleep(random())


    '''
    Internal API
    Clear port from previous experiments
    '''
    def clear_port(self):
        os.system('sudo fuser -k {}/tcp'.format(self.port))


    '''
    Internal API
    Listen for incoming messages from party members
    '''
    def listen(self):
        self.logger.info('listening on port %d' % self.port)

        try:
            while True:
                if self.ep_count == self.global_epochs:
                    self.logger.info('ps.listen teardown')
                    self.pc.teardown()
                    break

                conn, addr = self.sock.accept()
                self.logger.info('ps.listen from {}'.format(str(addr)))
                p = Process(target=self.receive, args=(conn, addr))
                p.start()

        except KeyboardInterrupt:
            self.logger('\nexiting...')
            sys.exit(0)


    '''
    Internal API
    This function is the main process that executes across the entire lifetime of the TCP connection.
    If this function exits, this peer will leave the entire hypergraph training session.

    Input: conn (tuple)
           add (tuple)
    '''
    def receive(self, conn, addr):
        try:
            resp = {}

            # Process that maintains a hyperedge
            while 1:
                if self.ep_count == self.global_epochs:
                    self.logger.info('ps.receive: training complete')
                    break

                packet = conn.recv(4096)

                # if peer closes connection, then remove that peer from PC connections
                if len(packet) == 0:
                    self.pc.remove('{}:{}'.format(addr[0], addr[1]))

                msg = packet.split('::')

                if len(msg) < 2:
                    self.logger.fatal('gs.receive(): empty message')
                    conn.sendall('invalid protocol')
                    return

                expected = int(msg[0])
                data = msg[1]

                while len(data) < expected:
                    # TODO change this to array join, string concat will get expensive if packets are large
                    data += conn.recv(min(expected - len(data), 4096))

                logging.info('ps.receive() addr:{}'.format(addr))
                resp = self.route({"addr": addr, "length": expected, "content": json.loads(data)})

                if resp == 'invalid':
                    self.logger.info('invalid message: ', data, ' from ', str(addr))

                conn.sendall(self.format_msg(resp))

        except ValueError as e:
            self.logger.fatal(e)
            conn.sendall('invalid protocol')
        finally:
            self.logger.info('closing connection')
            conn.close()


    '''
    Internal API
    Protocol for SMPL PS external API calls

    Input: msg (str)
    Output: str
    '''
    def format_msg(self, msg):
        msg = json.dumps(msg)
        return ''.join([str(len(msg)), '::', str(msg)])


    '''
    Internal API
    Map the incoming API call to the correct function

    Input:  msg (str)
    Output: response (str)
    '''
    def route(self, msg):

        content = msg['content']
        api = content['api']
        args = content['args'] if 'args' in content else None

        if api == 'establish_session':
            self.logger.info('api:establish_session')
            return self.establish_session(*args)
        elif api == 'synchronize_parameters':
            self.logger.info('api:synchronize_parameters')
            return self.synchronize_parameters(*args)
        elif api == 'get_parameters':
            self.logger.info('api:get_parameters')
            return self.get_parameters(*args)
        elif api == 'share_grad':
            self.logger.info('api:share_grad')
            return self.share_grad(*args)
        else:
            self.logger.info('api:{}'.format(api))
            return 'invalid'


    '''
    Internal API
    Wrapper function for train()
    '''
    def async_train(self):
        Process(target=self.train_hyperedge).start()


    '''
    Internal API
    Async establish clique and synchronize parameters
    '''
    def train_hyperedge(self):
        connected = False
        sess_id = 0

        # establish clique
        while not connected:
            connected, sess_id = self.init_session()
            sleep(1)

        self.logger.info('established hyperedge')

        # setup session in gradient queue
        self.grad_queue[sess_id] = {"peers":[], "gradients":[]}

        # each session should create its own model
        nn = TestNeuron() #NeuralNetwork()

        # pull current best parameters from parameter server to intialize model
        self.logger.info('updating nn params')
        best = json.loads(self.cache.get('best'))
        nn.update_parameters(best['parameters'])

        self.logger.info('training')
        self.train(sess_id, nn)

        ## LEFTOFF: checking 

        # compare recently trained hyperedge model with current best
        self.logger.info('update best model')
        self.update_model(sess_id)

        # clean up parameter cache and gradient queue
        self.cache.delete(sess_id)
        del self.grad_queue[sess_id]

        # increment total successful training epoches
        self.lock.acquire()
        self.ep_count += 1
        self.lock.release()


    '''
    Internal API

    Inputs:  sess_id (str) Session id
             nn      (NeuralNetwork) Neural Network
    Outputs:
    '''
    def train(self, sess_id, nn):
        '''
        - Hyper-parallelize with Hogwild!
        - Pass sess_id to Train so it can retrieve the session object from redis
        '''
        self.logger.info('training...')

        if self.parallel == 'hogwild':
            nn.share_memory()

        # DistributedTrainer constructor parameters
        conf = (sess_id, nn, self.allreduce, self.train_path, self.val_path, 
                1, self.cache, self.cuda, self.parallel, self.seed, False, True)
        processes = []

        for w in range(self.workers):
            p = Process(target=Train(conf).train)
            p.start()
            processes.append(p)

        # Locally synchronous
        for p in processes:
            p.join()


    '''
    Internal API
    Average gradients, distribute to other peers, and update gradients in model.
    This is only called by the Train.train() at the end of local training.

    Input:  sess_id (str)
            model   (NeuralNet)
            totla   (int) Total number of samples used to train. Required 
                          to calculate the weighted contribution of this 
                          peer's gradients
    '''
    def allreduce(self, sess_id, model, total):
        gradients = model.get_sparse_gradients(tolist=True)
        sess = json.loads(self.cache.get(sess_id))
        master = sess['master']

        if self.me == master:

            # wait until all peers have shared the gradients
            grad_queue = self.grad_queue[sess_id]
            while len(grad_queue['peers']) < len(sess['party']):
                pass

            # update model gradients
            model.add_batched_coordinates(grad_queue['gradients'], avg=self.clique+1)
            
            # update gradient queue so that self.share_grad() can reply with averaged gradients
            grad_queue['master_grad'] = model.get_gradients(tolist=True)
        
        else:
            send_to = master
            # if you're not master, send your gradients to master and wait for master to send you the average
            avg_grads = self.pc.send(send_to['host'], send_to['port'], 
                                    {"api": "share_grad", "args": [sess_id, self.me, gradients]})

            # update your model gradients with the average
            model.add_batched_coordinates(avg_grads)


    '''
    External API 
    Receive gradients from peers and reply with averaged gradients

    Inputs:  sess_id (str)    Session id
             peer    (dict)   Dictionary representing the sender
             gradients (list) Nested list of coordinate-gradient pairs
    Outputs:
    '''
    def share_grad(self, sess_id, peer, gradients):
        sess = json.loads(self.cache.get(sess_id))
        grad_queue = self.grad_queue[sess_id]
        grad_queue['peers'].append(peer['alias'])
        grad_queue['gradients'].extend(gradients)

        while 'master_grad' not in grad_queue:
            pass

        return grad_queue['master_grad']


    '''
    External API
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
            ok, resp = self.pc.send(best["host"], best["port"], {"api": "get_parameters", "args":["best"]})

            if len(resp) > 0:
                best_params = resp[0]
                self.cache.set(sess_id, json.dumps({"parameters": resp[0], "accuracy": resp[1], "val_size": 0, 
                                                    "train_size": 0, "peers": peers}))
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
    Internal API
    Initiates a hyperedge training session. This is only called from ps.train_hyperedge().

    Output: ok      (bool)
            sess_id (string)
    '''
    def init_session(self):
        ok = False
        sess_id = ''.join(['sess', str(getrandbits(randint(1,256)))])
        sess = {"id": sess_id, "me": self.me}
        resp = []

        peers = [x for x in self.establish_clique(sess) if len(x) > 0]

        # if can't connect with other peers, respond indicating failure
        if len(peers) < 1:
            # remove dead session from cache
            self.cache.delete(sess_id)
            return False, sess_id

        # sort by accuracy in descending order and cache as current session
        # implement parameter synchronization strategies here
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
        cond = best['alias'] != sess['me']['alias']
        if cond:
            ok, model = self.pc.send(best['host'], best['port'], {"api": "get_parameters", "args": ['best']})
        else:
            model = json.loads(self.cache.get('best'))

        # save parameters so can calculate difference (gradient) after training
        self.cache.set(sess_id, json.dumps({"parameters": model[0], "accuracy": model[1], "val_size": 0, "train_size": 0,
                                            "master": self.me, "party": peers, "pid": 0, "val": [0.0], "losses": []}))

        return ok, sess_id


    '''
    Get rank (float) of session model [0,100] where 
    100 is the lowest score and 0 is the highest

    Input:  sess_id (string) Session id
    '''
    def rank(self, sess_id):
        model = json.loads(self.cache.get(sess_id))
        model['rank'] = model['acc']*(self.val_rank/model['val_size'] + self.train_rank/model['train_size'])
        

    '''
    Internal API
    Update the accuracy and parameters of a session

    Input: sess_id    (str)    Session id
           parameters (tensor) Model parameters
           accuracy   (float)  Corresponding model accuracy
    '''
    def update_model(self, sess_id, parameters, accuracy):
        model = json.loads(self.cache.get(sess_id))

        if sess_id == 'best':
            if self.rank(sess_id) > model['rank']:
                model['parameters'] = parameters
                model['accuracy'] = accuracy
                self.cache.set(sess_id, json.dumps(model))
        else:
            model['parameters'] = parameters
            model['accuracy'] = accuracy
            self.cache.set(sess_id, json.dumps(model))


    '''
    Internal API
    Get all active sessions

    Output: list of tuples of sessions
    '''
    def active_sessions(self):
        active_ids = [k for k in self.cache.scan_iter('sess*')]

        if len(active_ids) == 0:
            return []

        sessions = self.cache.mget(active_ids)
        # [(<sess_id>, {"parameters": model[0], "accuracy": model[1], "party": peers}), ...]
        # peers = [{alias, host, port, accuracy}, ...]
        return zip(active_ids, sessions)


    '''
    Internal API
    Check that the given session does not overlap with the currently running sessions.
    If not existing cliques exist, then this returns an empty list.

    Input:  peers (list) List of dicts. See /distributed/config/party.json.
    Output: clique (list) List of dics. 
    '''
    def get_unique_clique(self, peers):
        possible_cliques = list(combinations(peers, self.clique))
        shuffle(possible_cliques)
        # [({u'alias': u'smpl-1', u'host': u'192.168.0.10', u'port': 9888, u'id': 1}, 
        #   {u'alias': u'smpl', u'host': u'192.168.0.12', u'port': 9888, u'id': 0})]

        clique = []
        active = self.active_sessions()
        #  [('sess1343003545191620262', '{"peers": [], "master": {"alias": "smpl-1", "host": "192.168.0.10", 
        #                                 "port": 9888, "id": 1}, "id": "sess1343003545191620262"}')]

        sess_hash = [hash(a) for a in active]

        if len(active) == 0:
            return list(possible_cliques.pop(0))

        # Check to ensure that overlapping cliques are not formed
        # Ensures that HDSGD forms a simple hypergraph
        while possible_cliques:
            clique = possible_cliques.pop(0)

            if hash(str(clique)) not in sess_hash:
                return list(clique)

        return clique


    '''
    Internal API
    Establish a training clique

    Input: sess (Session)
    Output: List of peers {alias, host, port, accuracy} to init_session()
    '''
    def establish_clique(self, sess):
        # find a unique clique
        unique = self.get_unique_clique(self.peers)
        peers = []

        # Note: Parallelize this!!!
        for send_to in unique:

            ok, resp = self.pc.send(send_to['host'], send_to['port'], 
                                    {"api": "establish_session", "args": [sess['id'], sess['me']]})

            if ok and len(resp) > 0:
                peers.append(resp)

            if len(peers) >= self.clique:
                break

        return peers[:self.clique]


    '''
    External API
    Reply to request to establish a session

    Input: sess_id (str), master (dict)
    Output: {alias, host, port, accuracy}
    '''
    def establish_session(self, sess_id, master):
        record = json.loads(self.cache.get('best'))
        
        me = dict(self.me)
        me['accuracy'] = record['accuracy']

        # CHECK FOR UNIQUE HYPEREDGES AGAIN AND IF A SESSION IN THE CACHE ALREADY HAS ALL THESE
        # PEERS EXACTLY, THEN ABORT THIS CURRENT SESSION

        self.cache.set(sess_id, json.dumps({"id": sess_id, "master": master, "peers": []}))
        return me


    '''
    External API
    API for getting this member's parameters

    Inputs: sess_id (str)
    Output: parameters (list) Nested list of lists
            accuracy (float)
    '''
    def get_parameters(self, sess_id):

        model = json.loads(self.cache.get(sess_id))
        
        if model == None:
            return [], -1

        # CALL DEEP GRADIENT COMPRESSION HERE
        if False:#self.sparsity > 0:
            model = json.loads(model)
            # return pt.largest_k(model['parameters'], self.sparsity), model['accuracy']
        else:
            return model['parameters'], model['accuracy']


    '''
    Internal API
    Input: signal, frame
    '''
    def force_stop(self, signal, frame):
        self.stop()


    '''
    Internal API
    Deletes everything in the redis cache
    '''
    def flush(self):
        self.cache.flushall()


    '''
    Internal API
    Stops the parameter server
    '''
    def stop(self):
        try:
            self.sock.close()
            self.logger.info('closing sockets')
            sys.exit(0)
        except Exception as e:
            self.logger.Fatal('Could not close ParameterServer socket')
            return False
        return True