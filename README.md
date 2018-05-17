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


#### Usage

```
usage: smpl.py [-h] [--host H] [--port P] [--cuda T] [--mpc T] [--infer T] [--save S] [--seed S] 
               [--log_freq L] [--name N] [-e E] [-i I] [-b B] [-m M] 
               [--lr L] [--nlr N] [-l L] [-s S] [-r T] [-t T] [-a A]

               
Simultaneous Multi-Party Learning Training

positional arguments:
-t --task           Tasks to perform
-a --arch           Architecture to use

optional arguments:
-h --help           Show help message
--host              Host address (default: 0.0.0.0)
--port              Port (default: 9090)
--cuda              Enables CUDA training (default: True)
--mpc               Use multi-party computation during training (default: True)
--infer             Set True for inference mode (default: False)
--save              Directory to save model parameters to (default: model/save)
--seed              Sets the seed for generating random numbers (default: 1)
--log_freq          Frequency for logging training (default: 100)
-n --name           Name for experiment
-e --epochs         Training epochs (default: 10)
-i --iterations     Training iterations (default: 3000)
-b --batch          Batch size (default: 64)
-m --momentum       Training momentum (default: 0.5)
--lr                Learning rate (default: 0.003)
--nlr               Learning rate of noisy gradients (default: 1e-02)
-l --limit          Share largest x-percent of gradients changed. Must set the mpc flag if using this. 
                    (default: 0.25)
-s --scale          Scale for preserving gradient accuracy during MPC (default: 10)
-r --replace        Replace corresponding gradients during training (default: True)
```