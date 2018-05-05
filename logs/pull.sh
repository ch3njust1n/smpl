#!/bin/bash
for server in $(cat ../server.txt)
do
        printf "\n$server:\n"
        scp -i ~/.ssh/smpl-moc.pem "$server":"/home/ubuntu/smpl/logs/*.log" .
done