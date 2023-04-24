touch fuse_configuration.cfg
echo "accountName $1">>fuse_configuration.cfg
echo "containerName demo" >>fuse_configuration.cfg
echo "authType SAS" >>fuse_configuration.cfg
echo "sasToken $2" >>fuse_configuration.cfg
echo "blobEndpoint https://$1.blob.core.windows.net/">>fuse_configuration.cfg

touch mount_storage.sh
echo "mkdir /mnt/blob_storage" >> mount_storage.sh
echo "chown gptaletest /mnt/blob_storage" >> mount_storage.sh
echo "mkdir /mnt/resource/blobfusetmp -p" >> mount_storage.sh
echo "chown gptaletest /mnt/resource/blobfusetmp" >> mount_storage.sh
echo "su -c \"blobfuse /mnt/blob_storage --tmp-path=/mnt/resource/blobfusetmp  --config-file=/home/gptaletest/fuse_configuration.cfg -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120\" gptaletest" >> mount_storage.sh

sh mount_storage.sh
