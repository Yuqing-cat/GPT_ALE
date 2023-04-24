#!/bin/bash

username=$(whoami)

echo "Mounting Blob Storage container for user: ${username}"

sudo mkdir /mnt/blob_storage
sudo chown ${username} /mnt/blob_storage

sudo mkdir /mnt/resource/blobfusetmp -p
sudo chown ${username} /mnt/resource/blobfusetmp

blobfuse /mnt/blob_storage --tmp-path=/mnt/resource/blobfusetmp  --config-file=/home/${username}/fuse_configuration.cfg -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120
