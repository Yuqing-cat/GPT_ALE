mkdir /mnt/blob_storage
chown <your_DSVM_username> /mnt/blob_storage

mkdir /mnt/resource/blobfusetmp -p
chown <your_DSVM_username> /mnt/resource/blobfusetmp

su -c "blobfuse /mnt/blob_storage --tmp-path=/mnt/resource/blobfusetmp  --config-file=/home/<your_DSVM_username>/fuse_configuration.cfg -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120" <your_DSVM_username>
