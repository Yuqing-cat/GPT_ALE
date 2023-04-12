wget -O ~/dbpedia_init_data.tar.gz https://palantirbackendstorage.blob.core.windows.net/demo/dbpedia_init_data.tar.gz
cd /mnt/blob_storage
tar -vxf ~/dbpedia_init_data.tar.gz
cp /mnt/blob_storage/dbpedia_14_dbpedia_14/annotated /mnt/blob_storage -r
#rm ~/dbpedia_init_data.tar.gz