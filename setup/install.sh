#!/bin/bash
# update system
sudo apt-get update
sudo apt-get upgrade

# Install Anaconda and create env named smpl
curl -O https://repo.continuum.io/archive/Anaconda2-5.0.1-Linux-x86_64.sh
mv Anaconda2-5.0.1-Linux-x86_64.sh /tmp
bash /tmp/Anaconda2-5.0.1-Linux-x86_64.sh
conda create --name smpl python=2.7

# Install dependencies
while read requirement; do conda install --yes $requirement || pip install $requirement; done < requirements.txt
sudo apt-get install redis-server
conda install pytorch-cpu torchvision -c pytorch
sudo apt install libgl1-mesa-glx

# Make smpl.py executable
chmod +x smpl.py

# Default activate Anaconda env
echo "export PATH=\"/home/ubuntu/anaconda2/bin:$PATH\"" >> ~/.bashrc
echo "source activate smpl" >> ~/.bashrc
echo "redis-server --daemonize yes" >> ~/.bashrc

# Make smpl executable
chmod +x smpl.py
chmod 755 smpl.py

source ~/.bashrc
