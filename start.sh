#!/bin/bash
for server in $(cat server.txt)
do
        printf "\n$server:\n"
        ssh -i ~/.ssh/smpl-0.pem "$server" "fuser -n tcp -k 9888; pkill -9 -f python;"
        ssh -f -i ~/.ssh/smpl-0.pem "$server" "redis-server --daemonize yes; cd smpl; chmod +x smpl.py; ./smpl.py"
done
./smpl.py