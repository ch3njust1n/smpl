#!/bin/bash
printf "killing port 9888\n"
for server in $(cat server.txt)
do
        printf "\n$server:\n"
        ssh -i ~/.ssh/smpl-0.pem "$server" "fuser -n tcp -k 9888; pkill -9 -f python"
done
fuser -n tcp -k 9888
pkill -9 -f python
