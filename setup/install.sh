#!/bin/bash
sudo apt-get update
while read requirement; do conda install --yes $requirement || pip install $requirement; done < requirements.txt
conda install pytorch-cpu torchvision -c pytorch
sudo apt install libgl1-mesa-glx
chmod +x smpl.py