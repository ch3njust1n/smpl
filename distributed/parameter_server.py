'''
    Justin Chen
    7.5.17

    Module for handling asynchronous gradient aggregation and sharing
'''

import sys
sys.path.insert(0, 'data')

from parameter_channel import ParameterChannel
from multiprocessing import Process, Lock, Manager, cpu_count, Value
from multiprocessing.managers import BaseManager
from threading import Thread
from random import random, getrandbits, shuffle, randint
from time import sleep, time
from train import Train
from datetime import datetime
from itertools import combinations
from model import network as net
import parameter_tools as pt
import os, torch, ujson, redis, socket, logging, utils, test, train, data


class ParameterServer(object):
    def __init__(self, args):
        
        self.ds_host        = args.ds_host
        self.ds_port        = args.ds_port
        self.host           = args.host
        self.port           = args.port
        self.workers        = (cpu_count()-args.clique)/(args.clique)

        self.async_global   = args.async_global
        self.async_mid      = args.async_mid
        self.async_local    = args.async_local
        self.batch_size     = args.batch_size
        self.clique         = args.clique-1
        self.cuda           = args.cuda
        self.data           = args.data
        self.dev            = args.dev
        self.drop_last      = args.drop_last
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
        self.shuffle        = args.shuffle
        self.sparsity       = args.sparsity
        self.strategy       = args.strategy
        self.train_rank     = args.train_rank
        self.val_rank       = args.val_rank

        self.__clear_port()

        # Locks
        self.edge_lock = Lock()
        self.count_lock = Lock()
        self.train_lock = Lock()

        # Get data
        Thread(target=self.__load_data).start()

        # Load party config
        self.peers = utils.load_json(os.path.join(os.getcwd(), 'distributed/config/', self.party))

        if len(self.peers) == 0:
            raise Exception('Error: party is empty')

        utils.check_party(self.peers)
        self.me = utils.get_me(self.peers, eth=self.eth)

        # For testing only so that we can see a difference in the parameters across peers
        self.seed = self.me['id']

        # Clear previous logs
        self.log_dir = os.path.join(os.getcwd(), 'logs')
        for file in os.listdir(self.log_dir):
            if file.endswith('.log'): os.remove(os.path.join(self.log_dir, file))

        self.log, self.log_path = utils.log(self.log_dir, 'ps{}'.format(self.me['id']))

        if self.dev:
            # CUDA/GPU settings
            if args.cuda and torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            else:
                torch.manual_seed(self.seed)

        # Save all state in Redis
        self.cache = redis.StrictRedis(host='localhost', port=6379, db=0)

        try:
            self.cache.ping()
        except ConnectionError:
            self.log.exception("Redis isn't running. try `sudo systemctl start redis`")
            exit(0)

        if args.flush:
            self.flush()

        # Setup parameter cache
        # Network() was named generically intentionally so that users can plug-and-play
        # Track best set of parameters. Equivalent of "global" params in central server model.
        # Stash this server's info
        # self.cache.set('best', ujson.dumps({"accuracy": 0.0, "val_size": 0, "train_size": 0, 
        #                                    "parameters": [x.data.tolist() for x in net.DevNet().parameters()]}))
        self.cache.set('best', ujson.dumps({"accuracy": 0.0, "val_size": 0, "train_size": 0, "log": self.log_path,
                                            "parameters": [x.data.tolist() for x in net.DevNet(self.seed, self.log).parameters()]}))
        self.cache.set('server', ujson.dumps({"clique": self.clique, "host": self.host, "port": self.port}))
        self.cache.set('edges', 0)
        self.cache.set('epochs', 0)

        # Setup TCP connections to all peers
        self.tcp_conn = {}
        self.pc = ParameterChannel(self.peers, logger=self.log)

        # Establish ports for receiving API calls
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(1)
        Thread(target=self.__listen).start()

        self.pc.setup()

        # Init training
        self.__async_train()


    '''
    Internal API
    Clear port from previous experiments
    '''
    def __clear_port(self):
        os.system('sudo fuser -k {}/tcp'.format(self.port))


    '''
    Internal API
    Listen for incoming messages from party members
    '''
    def __listen(self):
        self.log.info('listening on port %d' % self.port)

        try:
            while True:
                if int(self.cache.get('epochs')) == self.epochs:
                    self.log.info('ps.listen teardown')
                    self.pc.teardown()
                    break

                conn, addr = self.sock.accept()
                self.log.info('ps.listen from {}'.format(str(addr)))
                p = Process(target=self.__receive, args=(conn, addr))
                p.start()

        except KeyboardInterrupt:
            self.log.exception('\nexiting...')
            sys.exit(0)


    '''
    Internal API
    This function is the main process that executes across the entire lifetime of the TCP connection.
    If this function exits, this peer will leave the entire hypergraph training session.

    Input: conn (tuple)
           add (tuple)
    '''
    def __receive(self, conn, addr):
        try:
            resp = {}
            packet = ''
            # Process that maintains a hyperedge
            while 1:
                if int(self.cache.get('epochs')) == self.epochs:
                    self.log.info('ps.receive: training complete')
                    break

                packet = conn.recv(4096)
                msg = packet.split('::')

                if len(msg) < 2:
                    msg = 'empty message: {}, from {}'.format(packet, addr)
                    self.log.error(msg)
                    self.cache.set('empty{}'.format(time()), msg)
                    conn.sendall(self.__format_msg('invalid'))
                    return

                expected = int(msg[0])
                data = msg[1]

                while len(data) < expected:
                    # TODO change this to array join, string concat will get expensive if packets are large
                    data += conn.recv(min(expected - len(data), 4096))

                logging.info('ps.receive() addr:{}'.format(addr))
                try:
                    resp = self.__route({"addr": addr, "length": expected, "content": ujson.loads(data)})
                except ValueError as e:
                    raise Exception(e)

                if resp == 'invalid':
                    self.log.error('invalid message: {} from: {}'.format(data, str(addr)))

                conn.sendall(self.__format_msg(resp))

        except ValueError as e:
            self.log.exception(e)
            conn.sendall('invalid protocol')
        finally:
            self.log.info('closing connection')
            conn.close()


    '''
    Internal API
    Protocol for SMPL PS external API calls

    Input: msg (str)
    Output: str
    '''
    def __format_msg(self, msg):
        msg = ujson.dumps(msg)
        return '{}::{}'.format(len(msg), msg)


    '''
    Internal API
    Map the incoming API call to the correct function

    Input:  msg (str)
    Output: response (str)
    '''
    def __route(self, msg):

        content = msg['content']
        api = content['api']
        args = content['args'] if 'args' in content else None

        if api == 'establish_session':
            return self.__establish_session(*args)
        elif api == 'synchronize_parameters':
            return self.__synchronize_parameters(*args)
        elif api == 'get_parameters':
            return self.get_parameters(*args)
        elif api == 'share_grad':
            return self.__share_grad(*args)
        else:
            self.log.error('api:{}, args:{}'.format(api, args))
            return 'invalid'


    '''
    Internal API

    Request data from DataServer
    '''
    def __load_data(self):
        # self.pc.send(self.ds_host, self.ds_port, {"api": "get_data", "args":[self.me['alias']]})
        pass


    '''
    Internal API
    Wrapper function for train(). Continue to initiate hyperedges while
    your current hyperedge count is less than the specified max. Iterating in this fashion
    instead of spawning max processes at once and using join() allows self.cache.get('edges') to account
    for hyperedges created by other peers that this peer has joined else will cause a deadlock
    where no one joins anyone else's hyperedge and all peers request each other.
    '''
    # def __async_train(self):
    #     while int(self.cache.get('epochs')) < self.epochs:
    #         Process(target=self.train_hyperedge).start()
            
    #         self.train_lock.acquire()
    #         edge_count = int(self.cache.get('edges'))
    #         while edge_count == self.max:
    #             self.log.info('he_count: {}'.format(edge_count))
    #             self.train_lock.release()
    #             sleep(random())
    def __async_train(self):
        #### CREATE ONLY ONE HYPEREDGE FOR DEV ONLY
        Process(target=self.__train_hyperedge).start()


    '''
    Internal API
    Async establish clique and synchronize parameters
    '''
    def __train_hyperedge(self):
        log, log_path = utils.log(self.log_dir, '{}-{}'.format(self.me['id'], utils.get_date()))
        log.info('ps.__train_hyperedge')

        connected = False
        sess_id = ''

        # establish clique
        while len(sess_id) == 0:
            sess_id = self.__init_session(log=log, log_path=log_path)
            sleep(random())

        
        # Save log file path
        sess = ujson.loads(self.cache.get(sess_id))
        sess["log"] = log_path
        self.cache.set(sess_id, ujson.dumps(sess))

        sess = self.__train(sess_id, log)

        # compare recently trained hyperedge model with current best
        ok = self.update_model('best', session=sess_id, log=log)

        # clean up parameter cache and gradient queue
        if not self.dev:
            self.cache.delete(sess_id)

        # increment total successful training epoches and hyperedges
        self.count_lock.acquire()
        self.cache.set('epochs', int(self.cache.get('epochs'))+1)
        self.cache.set('edges', int(self.cache.get('edges'))-1)
        self.count_lock.release()
        log.info('hyperedge training complete')


    '''
    Internal API
    Initiates asynchronous local training and passes __allreduce() to each process
    so that each 

    Inputs:  sess_id (str) Session id
             nn      (NeuralNetwork) Neural Network
    '''
    def __train(self, sess_id, log=None):
        '''
        - Hyper-parallelize with Hogwild!
        - Pass sess_id to Train so it can retrieve the session object from redis
        '''
        if log == None:
            log, log_path = utils.log(self.log_dir, 'train-{}'.format(sess_id))


        # Setup variables for sharing gradients
        sess = ujson.loads(self.cache.get(sess_id))

        # Each session should create its own model
        nn = net.DevNet(seed=self.seed, log=log)
        nn.update_parameters(sess['parameters'])

        self.__local_train(sess_id, nn, log)

        # Update session model rank
        sess = ujson.loads(self.cache.get(sess_id))

        # Multi-step gradient between synchronized parameters and locally updated parameters
        multistep = nn.multistep_grad(sess['parameters'], sparsify=True)
        self.__allreduce(sess_id, sess, multistep, sess['train_size'], log)

        # Final validation
        # Retrieve gradients in session shared by peers
        sess = ujson.loads(self.cache.get(sess_id))
        log.debug('after\ntrain_size: {}, validation: {}'.format(sess['train_size'], sess['val_size']))
        sess['train_size'] += sess['share_train_sizes']
        nn.add_batched_coordinates(sess['gradients'], sess['train_size'])

        # Validate model accuracy
        conf = (log, sess_id, self.cache, nn, self.data, self.batch_size, self.cuda, self.drop_last, self.shuffle, self.seed)
        sess["accuracy"] = Train(conf).validate()
        self.cache.set(sess_id, ujson.dumps(sess))
        self.log.debug('after grad avg: {} ({}%)'.format(sess_id, sess["accuracy"]))

        return sess


    '''
    Function for initiating local training

    Input:  sess_id (string)
            nn      (nn.Module)
            log     (Logger)
    Output: share   (dict)      Dictionary containing accuracy, validation size, and training size
    '''
    def __local_train(self, sess_id, nn, log=None):
        # DistributedTrainer constructor parameters
        # network, sess_id, data, batch_size, cuda, drop_last, shuffle, seed
        self.seed=18
        conf = (log, sess_id, self.cache, nn, self.data, self.batch_size, self.cuda, self.drop_last, self.shuffle, self.seed)
        processes = []

        start_time = time()

        if self.async_local:
            log.info('hogwild!')
            nn.share_memory()

            for w in range(self.workers):
                p = Process(target=Train(conf).train)
                p.start()
                processes.append(p)

            # Local sync barrier
            for p in processes:
                p.join()
        else:
            log.info('vanilla')
            t = Train(conf)
            t.train()

        log.info('local ({} s)'.format(time()-start_time))


    '''
    Internal API
    Average gradients, distribute to other peers, and update gradients in model.
    This is only called by the Train.train() at the end of local training.

    Input:  sess_id     (str)  Session id
            gradients   (list) List of torch.FloatTensors
            sample_size (int)  Total number of samples used to train. Required 
                               to calculate the weighted contribution of this 
                               peer's gradients
    '''
    def __allreduce(self, sess_id, sess, gradients, sample_size, log=None):

        # Async send gradients to all peers in hyperedge
        for send_to in sess['party']:
            log.debug('sending to {}:{} train_size: {}, sess_id: {}'.format(send_to['host'], send_to['port'], sample_size, sess_id))
            Thread(target=self.pc.send, 
                   args=(send_to['host'], send_to['port'], 
                         {"api": "share_grad", "args": [sess_id, self.me['alias'], gradients, sample_size]})).start()

        log.debug('before\ntrain_size: {}, validation: {}'.format(sess['train_size'], sess['val_size']))
        # Wait until all peers have shared their gradients
        # Remove this barrier to make hyperedges asynchronous
        while 1:
            share_count = ujson.loads(self.cache.get(sess_id))['share_count']
            if int(share_count) == len(sess['party']): break
            if int(share_count) > len(sess['party']):
                log.debug('share_count: {} > sess[party]: {}'.format(int(share_count), len(sess['party'])))
                raise Exception('share count cannot be greater than total party size')
            sleep(random())
        self.log.debug('done sync barrier\ttrain_size: {}'.format(ujson.loads(self.cache.get(sess_id))['train_size']))


    '''
    External API 
    Receive gradients from peers.

    Inputs:  sess_id   (str)  Session id
             sender    (dict) Alias of sender
             gradients (list) Nested list of coordinate-gradient pairs
             samples   (int)  Number of samples sending peer used to generate given gradients
    Output:  ok        (bool) Bool indicating that session exists and values were updated
    '''
    def __share_grad(self, sess_id, sender, gradients, samples):
        if not self.cache.exists(sess_id):
            print 'sessDNE sess_id: {}'.format(sess_id)
            return False

        sess = ujson.loads(self.cache.get(sess_id))

        # Get log
        try:
            log_name = sess["log"]
            logging.basicConfig(filename=log_name, filemode='a', level=logging.DEBUG, datefmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            log = logging.getLogger()

            sess['share_count'] = 1 + int(sess['share_count'])
            sess['gradients'].append(gradients)
            log.debug('sess_id: {} samples:{} sess[train_size]:{}'.format(sess_id, samples, int(sess['train_size'])))
            sess['share_train_sizes'] = samples + int(sess['share_train_sizes'])
            self.cache.set(sess_id, ujson.dumps(sess))
        except KeyError as e:
            self.log.critical('KeyError: {}, sess: {}, sess_id: {}, gradients: {}'.format(e, sess, sess_id, gradients))
            return 'invalid'
        except Exception as e:
            raise Exception('Unexpected error: {}'.format(e))
            return 'invalid'

        return True


    '''
    External API
    Synchronize parameters

    Input:  sess_id    (str)  Unique session id
            best       (str)  Dict representing best peer
            peers      (list) List of dicts representing peers
            parameters (list) Nested list of lists containing model parameters
            accuracy   (int)  Accuracy associated with given model
    Output: ok         (bool) Flag indicating that sess importation was set in cache correctly
    '''
    def __synchronize_parameters(self, sess_id, best, peers, sender, parameters=[], accuracy=0):
        # Get log
        logname = ujson.loads(self.cache.get(sess_id))
        logging.basicConfig(filename=logname, filemode='a', level=logging.DEBUG, datefmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log = logging.getLogger()
        log.info('api:synchronize_parameters')

        ok = False
        peers.append(sender)

        # If not explicitely given parameters and accuracy, retrieve parameters from the specified best peer
        if len(parameters) == 0:
            # If my parameters do not have the best validation accuracy
            best_params = None
            sess = {}

            if best['host'] == self.me['host']:
                sess = ujson.loads(self.cache.get('best'))
                sess["party"] = peers
                parameters = sess["parameters"]
            else:
                # Always only wanna synchronize with the local parameters of peer with the best parameters
                # not their globally best parameters, just the parameters they're using for this hyperedge
                resp = []

                while len(resp) == 0:
                    _, resp = self.pc.send(best["host"], best["port"], {"api": "get_parameters", "args":[sess_id]})
                    sleep(random())
                ok = True
                parameters = resp[0]
                sess = {"parameters": parameters, "accuracy": resp[1], "val_size": 0, "train_size": 0, "party": peers}
            
            sess.update(ujson.loads(self.cache.get(sess_id)))
            ok = self.cache.set(sess_id, ujson.dumps(sess))
        else:
            # Else parameters were explicitely given, so update with those
            ok = self.update_model(sess_id, parameters=parameters, accuracy=accuracy, log=log)

        # Start locally training
        nn = net.DevNet(seed=self.seed, log=log)
        nn.update_parameters(parameters)
        Process(target=self.__train, args=(sess_id, log,)).start()

        return ok


    '''
    Internal API
    Initiates a hyperedge training session. This is only called from ps.train_hyperedge().

    Output: ok      (bool)
            sess_id (string)
    '''
    def __init_session(self, log=None, log_path=''):
        log.info('ps.__init_session')
        ok = False
        sess_id = ''.join(['sess', str(getrandbits(randint(1,256)))])
        sess = {"id": sess_id, "me": self.me}
        resp = []

        peers = [x for x in self.__establish_clique(sess, log=log) if len(x) > 0]

        # if can't connect with other peers, respond indicating failure
        if len(peers) == 0:
            log.debug('ps.__init_session: removing dead session')
            # remove dead session from cache
            self.cache.delete(sess_id)
            return ''

        # sort by accuracy in descending order and cache as current session
        # implement parameter synchronization strategies here
        # Note: Selecting based on best validation accuracy will lead to hyperedge collapse
        peers = sorted(peers, key=lambda x: x['accuracy'], reverse=True)
        best = peers[0] if random() <= self.epsilon else peers[randint(0, len(peers)-1)]

        # request parameters from member with highest accuracy
        model = []
        args = []

        if best['alias'] != sess['me']['alias']:
            ok, model = self.pc.send(best['host'], best['port'], {"api": "get_parameters", "args": ['best']})

            args = [sess_id, best, peers[:], self.me]
        else:
            model = ujson.loads(self.cache.get('best'))
            # send sess_id, parameters, and model accuracy to all peers
            args = [sess_id, best, peers[:], self.me, model[0], model[1]]

        log.debug('calling ps.synchronize_parameters sess_id: {}'.format(sess_id))

        # Synchronize parameters of model with best validation accuracy
        for i, send_to in enumerate(peers):
            args[2].pop(i)
            Thread(target=self.pc.send, 
                args=(send_to['host'], send_to['port'],
                      {"api": "synchronize_parameters", "args": args},)).start()

        log.debug('party:{}'.format(peers))

        # save parameters so can calculate difference (gradient) after training
        self.cache.set(sess_id, ujson.dumps({"parameters": model[0], "accuracy": model[1], "val_size": 0, 
                                            "train_size": 0, "party": peers, "pid": 0, "ep_losses": [],
                                            "log": log_path, "share_count": 0, "gradients": [], 
                                            "share_train_sizes": 0, "train_batches": 0, "val_batches": 0}))

        if not self.cache.exists(sess_id):
            log.error('Error: key insertion failure {}'.format(sess_id))
            raise Exception('Error: key insertion failure {}'.format(sess_id))

        return sess_id


    '''
    Internal API
    Get rank (float) of session model [0,100] where 
    100 is the lowest score and 0 is the highest

    Input:  sess_id (string) Session id
    '''
    def rank(self, sess_id, log=None):
        pass
        # model = ujson.loads(self.cache.get(sess_id))
        # model['rank'] = model['accuracy']*(self.val_rank/model['val_size'] + self.train_rank/model['train_size'])


    '''
    Internal API
    Update the accuracy and parameters of a session

    Input:  sess_id    (str)    Session id
            session    (string) Session id with updated values
            parameters (tensor) Model parameters
            accuracy   (float)  Corresponding model accuracy
            log        (Logger) Session log
    Output: ok         (bool)   Flag indicating if model was updated
    '''
    def update_model(self, sess_id, session='', parameters=[], accuracy=-1, log=None):
        log.debug('updating model id: {}'.format(sess_id))
        sess = ujson.loads(self.cache.get(sess_id))

        if len(session) > 0:
            update = ujson.loads(self.cache.get(session))

            if sess_id == 'best' and sess['accuracy'] > update['accuracy']:
                return False

            sess['parameters'] = update['parameters']
            sess['accuracy'] = update['accuracy']
            sess['val_size'] = update['val_size']
            sess['train_size'] = update['train_size']
        else:
            if sess_id == 'best' and sess['accuracy'] > accuracy:
                return False
            sess['parameters'] = parameters
            sess['accuracy'] = accuracy

        return self.cache.set(sess_id, ujson.dumps(sess))


    '''
    Internal API
    Get all active sessions

    Output: list of tuples of sessions
    '''
    def active_sessions(self, log=None):
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
    def get_unique_clique(self, peers, log=None):
        log.info('ps.get_unique_clique')
        possible_cliques = list(combinations(peers, self.clique))
        shuffle(possible_cliques)
        # [({u'alias': u'smpl-1', u'host': u'192.168.0.10', u'port': 9888, u'id': 1}, 
        #   {u'alias': u'smpl', u'host': u'192.168.0.12', u'port': 9888, u'id': 0})]

        clique = []
        active = self.active_sessions(log=log)
        #  [('sess1343003545191620262', '{"peers": [], "id": "sess1343003545191620262"}')]

        if len(active) == 0:
            return list(possible_cliques.pop(0))

        sess_hash = [hash(a) for a in active]

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
    Output: List of peers {alias, host, port, accuracy} to __init_session()
    '''
    def __establish_clique(self, sess, log=None):
        log.info('ps.__establish_clique')
        # find a unique clique
        unique = self.get_unique_clique(self.peers, log=log)
        peers = []
        responses = []

        # Note: Parallelize this!!!
        for send_to in unique:

            ok, resp = self.pc.send(send_to['host'], send_to['port'], 
                                    {"api": "establish_session", "args": [sess['id']]})

            if len(resp) > 0:
                peers.append(resp)

            if len(peers) >= self.clique:
                break

        return peers[:self.clique]


    '''
    External API
    Reply to request to establish a session

    Input: sess_id (str)
    Output: {alias, host, port, accuracy}
    '''
    def __establish_session(self, sess_id):
        # Setup logging for hyperedge
        log, log_path = utils.log(self.log_dir, '{}-{}'.format(self.me['id'], sess_id))
        log.info('api:establish_session')

        while not self.edge_lock.acquire():
            sleep(0.1)

        if int(self.cache.get('edges')) == self.max:
            log.debug('maxed hyperedges')
            return {}
        else:
            # Increment hyperedge count
            self.cache.set('edges', int(self.cache.get('edges'))+1)
            record = ujson.loads(self.cache.get('best'))
            
            me = dict(self.me)
            me['accuracy'] = record['accuracy']

            # CHECK FOR UNIQUE HYPEREDGES AGAIN AND IF A SESSION IN THE CACHE ALREADY HAS ALL THESE
            # PEERS EXACTLY, THEN ABORT THIS CURRENT SESSION
            self.cache.set(sess_id, ujson.dumps({"id": sess_id, "peers": [], "log": log_path, 
                                                 "share_train_sizes": 0, "share_count": 0, 
                                                 "gradients": []}))
        self.edge_lock.release()
        
        return me


    '''
    External API
    API for getting this member's parameters

    Inputs: sess_id (str)
    Output: parameters (list) Nested list of lists
            accuracy (float)
    '''
    def get_parameters(self, sess_id):
        try:
            log_name = ujson.loads(self.cache.get(sess_id))["log"]
            logging.basicConfig(filename=log_name, filemode='a', level=logging.DEBUG, datefmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            log = logging.getLogger()
            log.info('api:get_parameters sess_id: {}'.format(sess_id))

            model = ujson.loads(self.cache.get(sess_id))

            if model == None:
                return [], -1

            # CALL DEEP GRADIENT COMPRESSION HERE
            return model['parameters'], model['accuracy']
        except KeyError as e:
            self.log.exception('ps.get_parameters() sess_id:{}'.format(sess_id))
            return 'invalid'


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
            self.log.info('closing sockets')
            sys.exit(0)
        except Exception as e:
            self.log.exception('Could not close ParameterServer socket')
            return False
        return True