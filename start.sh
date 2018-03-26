#!/bin/bash
for server in $(cat server.txt)
do
        printf "\n$server:\n"
        ssh -i ~/.ssh/smpl-moc.pem "$server" "fuser -n tcp -k 9888; pkill -9 -f python; rm smpl/gradient.log"
        ssh -f -i ~/.ssh/smpl-moc.pem "$server" "cd smpl; ./smpl.py"
done
rm gradient.log 
clear
./smpl.py