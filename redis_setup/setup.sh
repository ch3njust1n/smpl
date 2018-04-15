#!/bin/bash
# Install Redis
sudo apt-get update
sudo apt-get install build-essential tcl
cd /tmp
curl -O http://download.redis.io/redis-stable.tar.gz
tar xzvf redis-stable.tar.gz
cd redis-stable
make
sudo make install
sudo mkdir /etc/redis
sudo cp redis.conf /etc/redis
sudo cp redis.service /etc/systemd/system

# Create the Redis User, Group and Directories
sudo adduser --system --group --no-create-home redis
sudo mkdir /var/lib/redis
sudo chown redis:redis /var/lib/redis
sudo chmod 770 /var/lib/redis

sudo cp redis.conf /etc/redis
sudo cp redis.service /etc/systemd/system
sudo systemctl start redis