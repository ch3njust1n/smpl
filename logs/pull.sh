#!/bin/bash
for server in $(cat ../server.txt)
do
        printf "\n$server:\n"
        scp -i ~/.ssh/smpl-0.pem "$server":"/home/ubuntu/smpl/logs/*.log" "/home/ubuntu/smpl/logs/"
done