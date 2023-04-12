username=`whoami`

touch fuse_configuration.cfg

echo "accountName $1">>fuse_configuration.cfg
echo "containerName demo" >>fuse_configuration.cfg
echo "authType SAS" >>fuse_configuration.cfg
echo "sasToken $2" >>fuse_configuration.cfg
echo "blobEndpoint https://$1.blob.core.windows.net/">>fuse_configuration.cfg

touch mount_storage.sh
echo "sudo mkdir /mnt/blob_storage" >> mount_storage.sh
echo "sudo chown ${username} /mnt/blob_storage" >> mount_storage.sh
echo "sudo mkdir /mnt/resource/blobfusetmp -p" >> mount_storage.sh
echo "sudo chown ${username} /mnt/resource/blobfusetmp" >> mount_storage.sh
echo "sudo blobfuse /mnt/blob_storage --tmp-path=/mnt/resource/blobfusetmp  --config-file=fuse_configuration.cfg -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120" >> mount_storage.sh

sh mount_storage.sh
