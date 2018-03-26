#!/bin/bash
ROOT_SMPL=~/smpl
ROOT_DEST=/home/ubuntu/smpl
DIST_DEST="$ROOT_DEST"/distributed
DIST_DIR="$ROOT_SMPL"/distributed
PS=parameter_server.py
PC=parameter_channel.py
PT=parameter_tools.py
MAIN=smpl.py

for server in $(cat server.txt)
do
        printf "\n$server:\n"
        scp -i ~/.ssh/smpl-moc.pem "$DIST_DIR"/"$PS" "$DIST_DIR"/"$PC" "$DIST_DIR"/"$PT" "$server":"$DIST_DEST"
        scp -i ~/.ssh/smpl-moc.pem "$MAIN" "$server":"$ROOT_DEST"
done
wait