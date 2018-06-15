# Simultaneous Multi-Party Learning

#### SMPL [sim-puh l] 
Hyper-parallel distributed training for deep neural networks


#### System Configuration

 - Be sure to set the correct IP address for all party members before running.
 - Check IPs in `server.txt`


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
cd smpl
bash start.sh
```

#### Diagnose Runs
```
python session.py --check -pl

If receiving issue about redis not importing even though it's installed, try deactivating and reactivating environment
```


#### Usage

```
usage: smpl.py [-h] [--host H] [--port P] [--cuda T] [--mpc T] [--infer T] [--save S] [--seed S] 
               [--log_freq L] [--name N] [-e E] [-i I] [-b B] [-m M] 
               [--lr L] [--nlr N] [-l L] [-s S] [-r T] [-t T] [-a A]

               
Simultaneous Multi-Party Learning Training

optional arguments:
--host                  Default host address (default: 0.0.0.0)
--port                  Port number for GradientServer
--async_global          Set for globally asynchronous training (default: True)
--async_mid             Set for asynchronous training within hyperedges (default: True)
--async_local           Set for asynchronous training on each peer (default: True)
--batch_size            Data batch size (default: 16)
--cuda                  Enables CUDA training (default: False)
-d --data               Data directory (default: mnist)
-v --dev                Development mode will fix random seed and keep session objects for analysis (default: True) 
--drop_last             True if last batch should be dropped if the dataset is not divisible by the batch size (default: False)
--ds_host               Data server host address
--ds_port               Data server port (default: 9888)
-x --epsilon            Chance of selecting a random set model during parameter synchronization. (default: 1.0)
--eth                   Peers ethernet interface (default: ens3)
-f --flush              Clear all parameters from previous sessions (default: True)
-e --hyperepochs        Total number of hyperepochs across all cliques for this peer (default: 1)
-l --local_parallel     Hogwild!, Divergent Exploration, or SGD (default: Hogwild!)
-lr --learning_rate     Learning rate (default: 1e-3)
--log_freq              Frequency for logging training (default: 100)
-n --name               Name of experiment (default: MNIST)
-p --party              Name of party configuration file. (default: party.json)
-v --regular            Maximum number of simultaneous hyperedges at any given time (default: 1)
-s --save               Directory to save trained model parameters to
--seed                  Random seed for dev only!
--shuffle               True if data should be shuffled (default: True)
--sparsity              Parameter sharing sparsification level (default: 0.0)
-u --uniform            Hyperedge size (default: 2)
--variety               Minimum number of new members required in order to enter into a new hyperedge. 
                        Prevents perfectly overlapping with current sessions. (default: 1)
```