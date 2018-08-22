# Simultaneous Multi-Party Learning

#### SMPL [sim-puh l] 
Hyper-parallel distributed training for deep neural networks


#### System Configuration
OpenStack Ubuntu 16.04 LTS
 - Be sure to set the correct IP address for all party members before running.
 - Check IPs in `server.txt`

#### Bash Scripts
You need to update the following bash scripts with the IPs of your nodes and your private key
- server.txt
- pull.sh
- start.sh
- stop.sh
- pulllogs.sh


#### Single Node Requirements Setup
```bash
Redis Setup
cd smpl/redis_setup
bash setup.sh
bash install.sh

The setup script should do this, but make smpl.py executable on all peers
chmod +x smpl.py

Check that `which python` matches the first line of `smpl.py`

cd bash/init
bash start.sh

Run `bash start.sh` from one of the nodes
```

#### Redis Server
````
If getting Could not connect to Redis at 127.0.0.1:6379: Connection refused enter the following on the appropriate node

redis-server --daemonize yes
````

#### Cluster Startup
```bash
bash sendall.sh
cd smpl #*** MUST CD into directory. Cannot do bash smpl/start.sh. Unsure why this happens
bash start.sh
```

#### Usage

```
Simultaneous Multi-Party Learning Training

optional arguments:
--host                  Host address (default: 0.0.0.0)
--port                  Port number for GradientServer (default: 9888)
--async_global          Set for globally asynchronous training (default: True)
--async_mid             Set for asynchronous training within hyperedges (default: True)
--async_local           Set for asynchronous training on each peer (default: True)
--batch_size            Data batch size (default: 16)
--cuda                  Enables CUDA training (default: False)
--data -d               Data directory (default: mnist)
--dev -v                Development mode will fix random seed and keep session objects for analysis (default: True) 
--drop_last             True if last batch should be dropped if the dataset is not divisible by the batch size (default: False)
--ds_host               Data server host address
--ds_port               Data server port (default: 9888)
--epsilon -x            Chance of selecting a random set model during parameter synchronization. (default: 1.0)
--eth                   Peers ethernet interface (default: ens3)
--flush -f              Clear all parameters from previous sessions (default: True)
--hyperepochs -e        Total number of hyperepochs across all cliques for this peer (default: 1)
--local_parallel -l     Hogwild!, Divergent Exploration, or SGD (default: Hogwild!)
--learning_rate -lr     Learning rate (default: 1e-3)
--log_freq              Frequency for logging training (default: 100)					
--name -n               Name of experiment (default: MNIST)
--party -p              Name of party configuration file. (default: party.json)
--regular -v            Maximum number of simultaneous hyperedges at any given time (default: 1)
--save -s               Directory to save trained model parameters to
--seed                  Random seed for dev only!
--smpc 					Use secure multiparty computation
--shuffle               True if data should be shuffled (default: True)
--sparsity              Parameter sharing sparsification level (default: 0.0)
--uniform -u            Hyperedge size (default: 2)
--variety               Minimum number of new members required in order to enter into a new hyperedge. 
                        Prevents perfectly overlapping with current sessions. (default: 1)
```

#### Diagnose Runs with session.py
```
Pull logs and display training summary
python session.py --check -pl

If receiving issue about redis not importing even though it's installed, try deactivating and reactivating environment

Get all redis objects
python session.py --keys

Examine a specific training session
python session.py -s sess289609693563518796840060 -m -p parameters
```

#### session.py usage

```
optional arguements:
--host                  Redis IP address (default: localhost)
--port                  Redis port number (default: 6379)
--case -c 				Set for case sensitive matching when using the --grep option
--check -ch 			Check that all hyperedges completed training
--clear 				Clear all logs
--database db 			Name of Redis database
--edges -e 				Display edge count
--grep - g 				Grep all files for given term
--ignore -i 			Ignores a particular key/value in the session object
--keys -k 				Get all Redis keys
--log_dir -l 			Log directory		
--minimal -m 			Ignore parameters and gradients
--sess -s 				Session objection id
--size -z 				Get size of cache object
--property -p 			Session object property
--properties -ps 		Get all properties of object
--pull -pl 				Pull logs from all peers
--variable -v 			Retrieve state variable. If using this, do not set --sess
```
