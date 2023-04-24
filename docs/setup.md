# Guidance to prepare your DSVM to run GPT-ALE

## Prerequisites

To run GPT-ALE, you need to have the following resources:

1. An Azure DSVM (recommended VM Size: "Standard NC6 Promo")
2. An Azure Storage Account

Please create these resources in the [Azure Portal](https://portal.azure.com/) before you proceed.

## Setup steps

Connect to your DSVM using SSH. We like to do this by using VS Code with the SSH extension.

### Clone GPT-ALE Repo

```
cd ~
git clone https://github.com/Azure/GPT_ALE
```

### 2. Setup Conda environment

To run GPT-ALE on the DSVM, we have configured all required dependency packages in a yml file, you can use this file to create an Anaconda environment. 

The configuration file can be found [here](../environment.yml)

To create the new conda environment, use following commands:
```
conda env create -f ~/GPT_ALE/environment.yml
```

After the above command finishes, activate the conda environment, all the following steps can be done in the conda environment.

```
conda activate gpt_ale
```


### 3. Mount the storage blob

> ``Note:`` The following instructions are for blobfuse version 1.0.2. If you are using a different version, or you don't have blobfuse installed yet, please refer to the [blobfuse documentation](https://learn.microsoft.com/en-us/azure/storage/blobs/storage-how-to-mount-container-linux).

We use blobfuse to mount a storage blob to local disk of the VM.

Please complete the following steps to mount the storage blob:

* Copy the file [fuse_configuration_template.cfg](../configs/fuse_configuration_template.cfg). 

`cp ./configs/fuse_configuration_template.cfg ./configs/fuse_configuration.cfg`

* Replace <your_storage_account_name>, <your_sas_token> with the account name of your blob storage account and your shared access signature of your container.

        Tips: remember to grant read AND write access for your SAS token

* Execute the script [scripts/mount_storage.sh](../scripts/mount_storage.sh), then replace <your_DSVM_username> with the real value.

* Mount the storage blob with following command:
```
sh configs/mount_storage.sh
```

> The script will prompt you for a "sudo password". This is simply the password you use to log into your DSVM.

After this, you can navigate to /mnt/blob_storage to verify the mount succeeded.

## Next steps

Return to the main [Readme](../README.md) and follow the steps under [Getting-Started](../README.md#getting-started)