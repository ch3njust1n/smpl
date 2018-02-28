# Simultaneous Multi-Party Learning

#### SMPL [sim-puh l] 
Hyper-parallel distributed training for deep neural networks


#### Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`

#### Basic usage

Be sure to set the correct IP address for all party members before running.
```bash
$ python smpl.py
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