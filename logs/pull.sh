#!/bin/bash
for server in $(cat ../server.txt)
do
        printf "\n$server:\n"
        rsync -e "ssh -i ~/.ssh/smpl-0.pem" -a "$server:/home/ubuntu/smpl/logs/*" "/home/ubuntu/smpl/logs/"
done