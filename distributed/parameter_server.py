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
from random import random, getrandbits, shuffle, randint, uniform
from time import sleep, time
from train import Train
from datetime import datetime
from itertools import combinations
from model import network as net
from data.dataset import Dataset
import parameter_tools as pt
import os, torch, ujson, redis, socket, logging, utils, test, train


class ParameterServer(object):
    def __init__(self, args, mode):
        # General parameters
        start_time = time()
        self.batch_size = args.batch_size
        self.cuda       = args.cuda
        self.data       = args.data
        self.drop_last  = args.drop_last
        self.eth        = args.eth
        self.log_level  = args.log_level
        self.party      = args.party
        self.seed       = args.seed
        self.shuffle    = args.shuffle
        self.workers    = cpu_count()

        # Load party config
        self.peers = utils.load_json(os.path.join(os.getcwd(), 'distributed/config/', self.party))

        if len(self.peers) == 0:
            raise Exception('Error: party is empty')

        utils.check_party(self.peers)
        self.me = utils.get_me(self.peers, eth=self.eth)

        self.log_dir = os.path.join(os.getcwd(), 'logs')
        self.log, self.log_path = utils.log(self.me['alias'], self.log_dir, 'ps{}'.format(self.me['id']), level=self.log_level)
        self.cache              = redis.StrictRedis(host='localhost', port=6379, db=0)

        try:
            self.cache.ping()
        except Exception:
            self.log.exception("Redis isn't running. try `sudo systemctl start redis`")
            exit(0)

        if args.flush:
            self.flush()

        self.log.info('Mode: {}'.format(mode))
        # Distributed training parameters
        if mode == 0:
            self.ds_host        = args.ds_host
            self.ds_port        = args.ds_port
            self.host           = args.host
            self.port           = args.port
            self.workers        = (cpu_count()-args.regular)/(args.regular)

            self.async_global   = args.async_global
            self.async_mid      = args.async_mid
            self.async_local    = args.async_local
            self.dev            = args.dev
            self.epsilon        = args.epsilon
            self.hyperepochs    = args.hyperepochs
            self.lr             = args.learning_rate
            self.name           = args.name
            self.parallel       = args.local_parallel
            self.regular        = args.regular
            self.save           = args.save
            self.sparsity       = args.sparsity
            self.uniform        = args.uniform
            self.uniform_ex     = self.uniform-1 # number of peers in an edge excluding itself
            self.variety        = args.variety

            self.__clear_port()

            # Locks
            self.best_lock = Lock()
            self.count_lock = Lock()
            self.done_lock = Lock()
            self.sock_lock = Lock()

            # For testing only so that we can see a difference in the parameters across peers
            if self.dev:
                self.seed = self.me['id']

            # Clear previous logs
            for file in os.listdir(self.log_dir):
                if file.endswith('.log'): os.remove(os.path.join(self.log_dir, file))

            if self.dev:
                # CUDA/GPU settings
                if args.cuda and torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.seed)
                else:
                    torch.manual_seed(self.seed)

            # Get data
            self.dataset = Dataset(self.cuda, self.batch_size, self.data, host=self.ds_host, port=self.ds_port)

            # Setup parameter cache
            # Network() was named generically intentionally so that users can plug-and-play
            # Track best set of parameters. Equivalent of "global" params in central server model.
            # Stash this server's info
            self.cache.set('best', ujson.dumps({"accuracy": 0.0, "val_size": 0, "train_size": 0, "log": self.log_path,
                                                "parameters": [x.data.tolist() for x in net.DevConv(self.seed, self.log).parameters()],
                                                "alias": self.me['alias']}))
            self.cache.set('server', ujson.dumps({"clique": self.uniform_ex, "host": self.host, "port": self.port}))
            self.cache.set('curr_edges', 0)
            self.cache.set('hyperedges', 0)
            self.cache.set('origin_edges', 0)
            self.cache.set('sock_pids', [])
            self.cache.set('done', 0)
            self.cache.set('peer_status', ujson.dumps([{'alias': p['alias'], 'done': 0} for p in self.peers]))

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
            self.log.info('total time: {} (seconds)'.format(time()-start_time))

            # Disconnect from hypergraph
            self.shutdown()
        else:
            # Get data
            self.dataset = Dataset(self.cuda, self.batch_size, self.data)

            # Setup model and train locally
            sess_id = self.get_id()
            nn = net.DevConv(seed=self.seed, log=self.log)
            parameters = [x.data.tolist() for x in nn.parameters()]
            self.log.info('depth: {}'.format(len(parameters)/2))
            self.cache.set(sess_id, ujson.dumps({"parameters": parameters, "accuracy": 0.0, "val_size": 0, 
                                                 "train_size": 0, "pid": 0, "ep_losses": [], "log": self.log_path, 
                                                 "gradients": [], "train_batches": 0, "val_batches": 0}))
            self.__local_train(sess_id, nn, async=args.async_local, log=self.log)


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
                conn, addr = self.sock.accept()

                if int(self.cache.get('hyperedges')) > len(self.peers):
                    self.log.info('reconnect from: {}'.format(addr))

                self.log.info('join from {}'.format(str(addr)))
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
        # Track pids for shutdown
        with self.sock_lock:
            sock_pids = ujson.loads(self.cache.get('sock_pids'))
            sock_pids.append(os.getpid())
            self.cache.set('sock_pids', ujson.dumps(sock_pids))

        try:
            resp = {}
            packet = ''
            # Process that maintains a hyperedge
            while 1:
                packet = conn.recv(4096)

                if len(packet) == 0:
                    # If this occurs, this process is dead and connection must be reset from the main PS process
                    self.log.info('{} closed socket'.format(addr))

                    # re-establish connection
                    self.pc.reconnect(addr)

                elif len(packet) < 2:
                    self.log.error('invalid message: {}, from {}'.format(packet, addr))
                    conn.sendall(self.pc.format_msg('invalid'))
                else:
                    msg = packet.split('::')

                    try:
                        expected = int(msg[0])
                        data = msg[1]
                    except ValueError as e:
                        self.log.error(msg)
                        conn.sendall(self.pc.format_msg('invalid'))

                    while len(data) < expected:
                        # TODO change this to array join, string concat will get expensive if packets are large
                        data += conn.recv(min(expected - len(data), 4096))

                    self.log.info('addr:{}'.format(addr))
                    try:
                        resp = self.__route({"addr": addr, "length": expected, "content": ujson.loads(data)})
                    except ValueError as e:
                        raise Exception(e)

                    if resp == 'invalid':
                        self.log.error('invalid message: {} from: {}'.format(data, str(addr)))

                    conn.sendall(self.pc.format_msg(resp))

        except ValueError as e:
            self.log.exception(e)
            conn.sendall('invalid protocol')
        except Exception as e:
            self.log.exception('other exception: {}'.format(e))
        finally:
            self.log.info('closing connection to {}'.format(addr))
            conn.close()


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
        elif api == 'done':
            return self.done(*args)
        else:
            self.log.error('api:{}, args:{}'.format(api, args))
            return 'invalid'


    '''
    Internal API

    Get a unique session id

    Output: sess_id (String) Session id
    '''
    def get_id(self):
        return ''.join(['sess', str(getrandbits(randint(1,256)))])


    '''
    Check if peer can support starting another hyperedge

    Input:  log (Logger, optional) Session log
    Output:     (bool)             True if this peer can start another hyperepoch
    '''
    def available(self, log=None):
        return int(self.cache.get('curr_edges')) < self.regular


    '''
    Internal API
    Wrapper function for train(). Continue to initiate hyperedges while
    your current hyperedge count is less than the specified max. Iterating in this fashion
    instead of spawning max processes at once and using join() allows self.cache.get('curr_edges') to account
    for hyperedges created by other peers that this peer has joined else will cause a deadlock
    where no one joins anyone else's hyperedge and all peers request each other.
    '''
    def __async_train(self):
        procs = [Process(target=self.__train_hyperedge) for i in range(0, self.hyperepochs)]

        while 1:
            sleep(uniform(3,5))
            
            if len(procs) > 0:
                with self.count_lock:
                    if self.available():
                        try:
                            self.done_lock.acquire()
                            procs.pop().start()
                            self.cache.set('origin_edges', self.hyperepochs-len(procs))
                            self.log.info('origin_edges: {}'.format(self.hyperepochs-len(procs)))
                            self.done_lock.release()
                        except IndexError:
                            self.log.info('max hyperepochs')
            else:
                # check for all completion boardcasts
                with self.done_lock:
                    peers = ujson.loads(self.cache.get('peer_status'))
                    done_count = sum([int(p['done']) for p in peers])

                    if len(peers) == done_count:
                        break
                        
        self.log.info('hypergraph complete')


    '''
    Internal API
    Async establish clique and synchronize parameters
    '''
    def __train_hyperedge(self):
        start = time()
        log, log_path = utils.log(self.me['alias'], self.log_dir, '{}-origin-{}'.format(self.me['id'], utils.get_date()), level=self.log_level)
        log.info('train_hyperedge')

        connected = False
        sess_id = ''

        while len(sess_id) == 0:
            # establish clique
            sleep(uniform(0,3))
            sess_id = self.__init_session(log=log, log_path=log_path)

        log.info('session id: {}'.format(sess_id))

        with self.count_lock:
            self.cache.set('curr_edges', int(self.cache.get('curr_edges'))+1)

        # Save log file path
        sess = ujson.loads(self.cache.get(sess_id))
        sess["log"] = log_path
        self.cache.set(sess_id, ujson.dumps(sess))

        self.__train(sess_id, log)
        log.info('hyperedge time: {} (seconds)'.format(time()-start))


    '''
    Internal API
    Initiates asynchronous local training and passes __allreduce() to each process
    so that each 

    Inputs:  sess_id (str) Session id
             log     (Logger, optional) Session log
    '''
    def __train(self, sess_id, log=None):
        '''
        - Hyper-parallelize with Hogwild!
        - Pass sess_id to Train so it can retrieve the session object from redis
        '''
        if log == None:
            log, log_path = utils.log(self.me['alias'], self.log_dir, 'train-{}'.format(sess_id))

        log.info('training...')

        # Setup variables for sharing gradients
        sess = ujson.loads(self.cache.get(sess_id))

        # Each session should create its own model
        nn = net.DevConv(seed=self.seed, log=log)
        nn.update_parameters(sess['parameters'])
        conf = (log, sess_id, self.cache, nn, self.dataset, self.batch_size, self.cuda, self.drop_last, self.shuffle, self.seed)

        self.__local_train(sess_id, nn, conf, async=self.async_local, log=log)

        # Update session model rank
        sess = ujson.loads(self.cache.get(sess_id))
        log.info('acc before allreduce: {}'.format(sess["accuracy"]))

        # Multi-step gradient between synchronized parameters and locally updated parameters
        multistep = nn.multistep_grad(sess['parameters'], k=self.sparsity, sparsify=True)
        self.__allreduce(sess_id, sess, multistep, sess['train_size'], log=log)

        # Final validation
        # Retrieve gradients in session shared by peers
        sess = ujson.loads(self.cache.get(sess_id))
        sess['train_size'] += sess['share_train_sizes']
        log.debug('addgrads: {}'.format(sess['gradients']))
        nn.add_batched_coordinates(sess['gradients'], lr=self.lr, avg=sess['train_size'])

        # Validate model accuracy
        # conf = (log, sess_id, self.cache, nn, self.dataset, self.batch_size, self.cuda, self.drop_last, self.shuffle, self.seed)
        sess["accuracy"] = Train(conf).validate()
        sess["done"] = True
        self.cache.set(sess_id, ujson.dumps(sess))

        # compare recently trained hyperedge model with current best
        self.update_best(sess_id, log=log)
        self.cleanup(sess_id, sess, log=log)
        self.broadcast(sess, log=log)

        return sess


    '''
    Function for initiating local training

    Input:  sess_id (string)           Session id
            nn      (nn.Module)        Neural network
            conf    (Tuple)            Tuple containing training variables
            log     (Logger, optional) Session log
    Output: share   (dict)             Dictionary containing accuracy, validation size, and training size
    '''
    def __local_train(self, sess_id, nn, conf, async=False, log=None):
        # DistributedTrainer constructor parameters
        # network, sess_id, data, batch_size, cuda, drop_last, shuffle, seed
        # conf = (log, sess_id, self.cache, nn, self.dataset, self.batch_size, self.cuda, self.drop_last, self.shuffle, self.seed)
        processes = []

        start_time = time()

        if async:
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
            Train(conf).train()

        log.info('local ({} s)'.format(time()-start_time))


    '''
    Internal API

    Update the best model on this parameter server, remove finished sessions from cache, 
    and update variable counters

    Input: sess_id (string)           Session id
           sess    (dict)             Session object
           log     (Logger, optional) Session log
    '''
    def cleanup(self, sess_id, sess, log=None):
        
        # clean up parameter cache and gradient queue
        if not self.dev:
            self.cache.delete(sess_id)

        # increment total successful training epoches and hyperedges
        with self.count_lock:
            total = int(self.cache.get('hyperedges'))
            self.cache.set('hyperedges', total+1)
            count = int(self.cache.get('curr_edges'))

            if count > 0:
                self.cache.set('curr_edges', count-1)

            log.debug('hyperedge complete\tcurr_edges:{}\thyperepochs: {}'.format(count, total))


    '''
    Internal API
    Initiates a hyperedge training session. This is only called from ps.train_hyperedge().

    Input:  log      (Logger, optional) Session log
            log_path (string, optional) Absolute path to session log
    Output: sess_id  (string) Session id or empty string if could not establish a session
    '''
    def __init_session(self, log=None, log_path=''):
        ok = False
        sess_id = self.get_id()
        sess = {"id": sess_id, "me": self.me}
        resp = []

        peers = [x for x in self.__establish_clique(sess, log=log) if len(x) > 0]

        # if can't connect with other peers, respond indicating failure
        if len(peers) == 0:
            self.kill_session(sess_id, peers)
            return ''

        # sort by accuracy in descending order and cache as current session
        # implement parameter synchronization strategies here
        # Note: Selecting based on best validation accuracy will lead to hyperedge collapse
        model = {}
        args = []
        all_peers = list(peers)
        parameters = []
        accuracy = -1

        with self.best_lock:
            model = ujson.loads(self.cache.get('best'))
            parameters = model["parameters"]
            accuracy = model["accuracy"]

        my_best = dict(self.me)
        my_best['accuracy'] = accuracy

        all_peers.append(my_best)
        all_peers = sorted(all_peers, key=lambda x: x['accuracy'], reverse=True)
        best = all_peers[0] if random() <= self.epsilon else all_peers[randint(0, len(all_peers)-1)]
        
        # request parameters from member with highest accuracy
        if best['alias'] != sess['me']['alias']:
            ok, model = self.pc.psend(best['host'], best['port'], {"api": "get_parameters", "args": ['best']})

            if len(model) == 0 or len(model[0]) == 0:
                self.kill_session(sess_id, peers)
                return ''

            parameters = model[0]
            accuracy = model[1]
            args = [sess_id, best, peers[:], self.me]
        else:
            # send sess_id, parameters, and model accuracy to all peers
            args = [sess_id, best, peers[:], self.me, parameters, accuracy]

        try:
            # save parameters so can calculate difference (gradient) after training
            self.cache.set(sess_id, ujson.dumps({"parameters": parameters, "accuracy": accuracy, "val_size": 0, 
                                                "train_size": 0, "party": peers, "pid": 0, "ep_losses": [],
                                                "log": log_path, "share_count": 0, "gradients": [], 
                                                "share_train_sizes": 0, "train_batches": 0, "val_batches": 0,
                                                "done": False, "type": 1}))
        except (IndexError, KeyError) as e:
            log.error(e)

        if not self.cache.exists(sess_id):
            log.error('Error: key insertion failure {}'.format(sess_id))
            self.kill_session(sess_id, peers)
            return ''

        # Synchronize parameters of model with best validation accuracy
        for i, send_to in enumerate(peers):
            Thread(target=self.pc.psend, 
                args=(send_to['host'], send_to['port'],
                      {"api": "synchronize_parameters", "args": args},)).start()

        return sess_id


    '''
    Internal API
    Update a session object with multiple key-value pairs

    Input:  sess_id (string) Session id
            data    (dict)   Dictionary to extend session
    Output: ok      (bool)   True if successfully updated the session, else False
    '''
    def extend_model(self, sess_id, data, log=None):
        ok = False

        if sess_id == 'best' or not self.cache.exists(sess_id):
            log.error('invalid session id: {}'.format(sess_id))
            return ok, {}

        sess = ujson.loads(self.cache.get(sess_id))
        sess.update(data)

        return self.cache.set(sess_id, ujson.dumps(sess)), sess


    '''
    Internal API
    Update the accuracy and parameters of a session. This function should not be used to update 
    the best sessions

    Input:  sess_id (string)           Session id
            key     (string)           Property to update or add
            value   (string)           Corresponding key's value
            log     (Logger, optional) Session log
    Output: ok      (bool)             Flag indicating if model was updated
    '''
    def update_model(self, sess_id, key, value, log=None):
        ok = False

        if sess_id == 'best' or not self.cache.exists(sess_id):
            log.error('invalid session id: {}'.format(sess_id))
            return ok, {}

        if key == None and value == None:
            log.error('invalid key: {}, value: {}'.format(key, value))
            return ok, {}

        sess = ujson.loads(self.cache.get(sess_id))
        sess[key] = value

        return self.cache.set(sess_id, ujson.dumps(sess)), sess


    '''
    Internal API
    Update the best model

    Input:  sess_id (string)           Session id for session that will be used to update the target session
            log     (Logger, optional) Session log
    Output: ok      (bool)             Flag indicating if model was updated
    '''
    def update_best(self, sess_id, log=None):
        ok = False

        if not self.cache.exists(sess_id):
            return ok, {}

        with self.best_lock:
            best = ujson.loads(self.cache.get('best'))
            log.debug('before updating best: {}'.format(best['accuracy']))

        update = ujson.loads(self.cache.get(sess_id))

        if update['accuracy'] <= best['accuracy']:
            return ok, {}

        best['accuracy'] = update['accuracy']
        best['val_size'] = update['val_size']
        best['parameters'] = update['parameters']
        best['train_size'] = update['train_size']

        with self.best_lock:
            log.debug('after updating best: {}'.format(best['accuracy']))
            ok = self.cache.set('best', ujson.dumps(best))

        return ok, best


    '''
    Internal API
    Get all active sessions

    Input:  log   (Logger, optional) Session log
    Output: list of tuples of sessions
    '''
    def active_sessions(self, log=None):
        active_ids = [k for k in self.cache.scan_iter('sess*')]    

        if len(active_ids) == 0:
            return []

        sessions = self.cache.mget(active_ids)

        # In inference mode, sessions will delete themselves when complete. But in dev mode,
        # can't tell the diference between active and inactive sessions because completed sessions
        # do no remove themselves from the redis cache
        if self.dev:
            active = []
            for i, sess in enumerate(sessions):
                if ujson.loads(sess)['done']:
                    active.append((active_ids[i], sess))

            return active
        else:
            # [(<sess_id>, {"parameters": model[0], "accuracy": model[1], "party": peers}), ...]
            # peers = [{alias, host, port, accuracy}, ...]
            return zip(active_ids, sessions)


    '''
    Internal API
    Check that the given session does not overlap with the currently running sessions.
    If not existing cliques exist, then this returns an empty list.

    Input:  peers  (list)             List of dicts. See /distributed/config/party.json.
            log    (Logger, optional) Session log
    Output: clique (list)             List of dicts. 
    '''
    def get_unique_clique(self, peers, log=None):
        possible_cliques = list(combinations(peers, self.uniform_ex))
        shuffle(possible_cliques)

        clique = []
        active = self.active_sessions(log=log)

        # If no active hyperedges, return random hyperedge
        if len(active) == 0:
            return list(possible_cliques.pop(0))

        active_edges = [ujson.loads(a[1])['party'] for a in active]
        shuffle(active_edges)

        # Check to ensure that overlapping cliques are not formed
        # Ensures that HDSGD forms a simple hypergraph
        for possible in possible_cliques:
            possible_set = set([str(p) for p in possible])
            intersect_lens = [len(possible_set.intersection(set([str(e) for e in edge]))) for edge in active_edges]
                        
            if self.uniform - max(intersect_lens) >= self.variety:
                return list(possible)

        return clique


    '''
    Internal API
    Establish a training clique

    Input:  sess (dict)             Session object
            log  (Logger, optional) Session log
    Output: List of peers {alias, host, port, accuracy} to __init_session()
    '''
    def __establish_clique(self, sess, log=None):
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

        return peers


    '''
    Internal API
    Average gradients, distribute to other peers, and update gradients in model.
    This is only called by the Train.train() at the end of local training.

    Input:  sess_id     (str)  Session id
            sess        (dict) Session object
            gradients   (list) List of torch.FloatTensors
            sample_size (int)  Total number of samples used to train. Required 
                               to calculate the weighted contribution of this 
                               peer's gradients
            log         (Logger, optional) Session log
    '''
    def __allreduce(self, sess_id, sess, gradients, sample_size, log=None):

        # Async send gradients to all peers in hyperedge
        try:
            for send_to in sess['party']:
                log.info('sending to {}:{} sess_id: {}'.format(send_to['host'], send_to['port'], sess_id))
                Thread(target=self.pc.psend, 
                       args=(send_to['host'], send_to['port'], 
                             {"api": "share_grad", "args": [sess_id, self.me['alias'], gradients, sample_size]})).start()
        except KeyError as e:
            log.error('error: {}, keys: {}'.format(e, sess.keys()))

        # Wait until all peers have shared their gradients
        # Remove this barrier to make hyperedges asynchronous
        while 1:
            sess = ujson.loads(self.cache.get(sess_id))
            share_count = sess['share_count']

            if int(share_count) == len(sess['party']): break
            if int(share_count) > len(sess['party']):
                log.error('share count ({}) cannot be greater than total party size ({})'.format(int(share_count), len(sess['party'])))
            sleep(random())

        log.info('done sync barrier')


    '''
    Internal API

    Input: sess (dict)   Session object
           log  (Logger, optional) Session log
    '''
    def broadcast(self, sess, log=None):
        # Check if this is the last hyperedge spawned by this ParameterServer
        # and set status indicating that all spawned hyperedges successfully finished
        if sess['type'] == 1 and int(self.cache.get('done')) == 0:
            with self.done_lock:
                orig = int(self.cache.get('origin_edges'))
                if orig == self.hyperepochs:
                    self.cache.set('done', 1)

                    # Broadcast to all peers that you're done

                    msgs = []
                    for send_to in self.peers:
                        t = Thread(target=self.pc.psend, args=(send_to['host'], send_to['port'], {"api": "done", "args": [self.me['alias']]}))
                        msgs.append(t)
                        t.start()

                    for t in msgs: t.join()

                    log.info('broadcasting complete')


    '''
    Internal API

    Broadcast to pre-emptively end a session
    Input:  sess_id (string) Session id
            peers   (list)   List of peers in the corresponding session 
    '''
    def kill_session(self, sess_id, peers):
        for send_to in peers:
            Thread(target=self.pc.psend, args=(send_to['host'], send_to['port'], {"api": "preempt", "args": [sess_id]})).start()


    '''
    External API
    Reply to request to establish a session

    Input: sess_id (str)
    Output: {alias, host, port, accuracy}
    '''
    def __establish_session(self, sess_id):
        # Setup logging for hyperedge
        log_name = '{}-{}'.format(self.me['id'], sess_id)
        log, log_path = utils.log(self.me['alias'], self.log_dir, log_name, mode='w', level=self.log_level)
        log.info('api:establish_session')

        with self.count_lock:

            if not self.available(log=log):
                os.remove(log_path)

                return {}
            else:
                log.info('starting hyperedge: {}'.format(sess_id))
                # Increment hyperedge count
                self.cache.set('curr_edges', int(self.cache.get('curr_edges'))+1)

                record = {}
                with self.best_lock:
                    record = ujson.loads(self.cache.get('best'))
                
                me = dict(self.me)
                me['accuracy'] = record['accuracy']
                log.debug('initiating with best: {}'.format(me['accuracy']))

                self.cache.set(sess_id, ujson.dumps({"id": sess_id, "log": log_path,
                                                     "share_train_sizes": 0, "share_count": 0, 
                                                     "gradients": [], "done": False, "type": 0}))
                return me


    '''
    External API
    Synchronize parameters

    Input:  sess_id    (str)            Unique session id
            best       (str)            Dict representing best peer
            peers      (list)           List of dicts representing peers
            parameters (list, optional) Nested list of lists containing model parameters
            accuracy   (int, optional)  Accuracy associated with given model
    Output: ok         (bool)           Flag indicating that sess importation was set in cache correctly
    '''
    def __synchronize_parameters(self, sess_id, best, peers, sender, parameters=[], accuracy=0):
        # Get log
        log_name = ujson.loads(self.cache.get(sess_id))["log"]
        log, log_path = utils.log(self.me['alias'], self.log_dir, log_name)
        log.info('api:synchronize_parameters')

        # Remove itself from its own peer list
        for i, p in enumerate(peers):
            if p['host'] == self.me['host']:
                peers.pop(i)
                break

        ok = False
        peers.append(sender)
        sess = {}

        # If not explicitely given parameters and accuracy, retrieve parameters from the specified best peer
        if len(parameters) == 0:
            # If my parameters do not have the best validation accuracy
            best_params = None
            sess = {}

            if best['host'] == self.me['host']:
                with self.best_lock:
                    sess = ujson.loads(self.cache.get('best'))
                    sess["party"] = peers
                    parameters = sess["parameters"]
                    log.debug('synchronizing with my best: {}'.format(sess['accuracy']))
            else:
                # Always only wanna synchronize with the local parameters of peer with the best parameters
                # not their globally best parameters, just the parameters they're using for this hyperedge
                resp = []
                while len(resp) == 0:
                    _, resp = self.pc.psend(best["host"], best["port"], {"api": "get_parameters", "args":[sess_id]})
                    sleep(random())

                ok = True
                parameters = resp[0]
                sess = {"parameters": parameters, "accuracy": resp[1], "val_size": 0, "train_size": 0, "party": peers}

            sess.update(ujson.loads(self.cache.get(sess_id)))
            ok = self.cache.set(sess_id, ujson.dumps(sess))
        else:
            # Else parameters were explicitely given, so update with those
            ok, sess = self.extend_model(sess_id, {"accuracy": accuracy, "parameters": parameters, "party": peers}, log=log)

        try:
            log.debug('party: {}'.format(sess['party']))
        except KeyError:
            log.error('No party key, len(sess): {}'.format(len(sess)))

        # Start locally training
        nn = net.DevConv(seed=self.seed, log=log)
        nn.update_parameters(parameters)
        Process(target=self.__train, args=(sess_id, log,)).start()

        return ok


    '''
    External API
    API for getting this member's parameters

    Inputs: sess_id    (str)   Session id
    Output: parameters (list)  Nested list of lists
            accuracy   (float) Parameter accuracy or -1 if cannot load parameters
    '''
    def get_parameters(self, sess_id):
        if not self.cache.exists(sess_id):
            print 'sessDNE sess_id: {}'.format(sess_id)
            return [], -1

        log_name = ujson.loads(self.cache.get(sess_id))["log"]
        log, log_path = utils.log(self.me['alias'], self.log_dir, log_name)
        log.info('api:get_parameters sess_id: {}'.format(sess_id))

        model = ujson.loads(self.cache.get(sess_id))

        if model == None:
            return [], -2

        # CALL DEEP GRADIENT COMPRESSION HERE
        return model['parameters'], model['accuracy']



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
            log, log_path = utils.log(self.me['alias'], self.log_dir, sess["log"])

            sess['share_count'] = 1 + int(sess['share_count'])
            sess['gradients'].append(gradients)
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

    Only called by another peer from their allreduce() if they finished training the 
    self.hyperepochs-amount of sessions they started

    Input: sender (string) Alias of peer
    '''
    def done(self, sender):
        with self.done_lock:
            peers = ujson.loads(self.cache.get('peer_status'))
            i = next((i for (i, d) in enumerate(peers) if d['alias'] == sender), None)
            peers[i]['done'] = 1
            self.cache.set('peer_status', ujson.dumps(peers))


    '''
    External API

    Preemptively end a session
    Input:  sess_id (string) Session id
    Output: ok      (bool)   True if curr_edges count was updated successfully
    '''
    def preempt(self, sess_id):
        ok = False
        with self.count_lock:
            self.cache.delete(sess_id)
            ok = self.cache.set('curr_edges', int(self.cache.get('curr_edges'))-1)
        return ok


    '''
    Internal API
    Input: signal, frame
    '''
    def force_stop(self, signal, frame):
        self.shutdown()


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
    def shutdown(self):
        try:
            self.pc.teardown()
            self.log.info('exiting ps')
            [os.kill(pid, signal.SIGTERM) for pid in ujson.loads(self.cache.get('sock_pids'))]
            sys.exit(0)
        except Exception as e:
            self.log.error(e)
            return False
        return True