# Guidance to prepare your DSVM to run GPT_ALE

## Prerequisites

Finish the step in [Here](./provision_azure_resources.md) and log in to your DSVM.
    Tips: default vm user name is "gptaletest"

## Setup steps

### 1. Clone GPT_ALE Repo

```
cd ~
git clone https://github.com/Azure/GPT_ALE
cd GPT_ALE
```

### 2. Setup conda environment
To run GPT_ALE on the DSVM, we have configured all required dependent packages in a yml file, you can use this file to create an Anaconda environment. 

The configuration file can be found at [Here](../environment.yml)

To create a new conda environment, use following commands:
```
conda env create -f environment.yml
```

After the above command finishes (this may take a few minutes), activate the conda environment, all the following steps can be done in the conda environment.

```
conda activate gpt_ale
```

### 3. Mount the storage blob

We use blobfuse to mount a stoarge blob to local disk of the VM. 
To get this ready, 3 steps are required:
* Automatic mount

    Use following commands to automatically mount the blob.
        
        cd ~
        sudo sh ~/GPT_ALE/docs/setup_files/mount_blob.sh <your storage account name> '<you sas token>'
        
* Manual mount
    * Create fuse_configuration.cfg at /home/<your_dsvm_user_name> (you can also use ~ instead.), copy the content of [fuse_configuration.cfg](./setup_files/fuse_configuration.cfg), then replace <your_storage_account_name>, <your_sas_token> with the real value.

        > Remember to grant **read**, **add**, **create**, **delete**, **write** and **list** access for your SAS token

    * Create mount_storage.sh at ~ directory, copy and paste the content of [mount_storage.sh](./setup_files/mount_storage.sh), then replace <your_DSVM_username> with the real value.

    * Mount the storage blob with following command:
    ```
    sudo sh ~/mount_storage.sh
    ```

After this, you can navigate to /mnt/blob_storage to verify the mount is succeeded.

### 4. Prepare storage container

After the blob has been mounted, run following script to prepare data for your storage.

```
sh ~/GPT_ALE/docs/setup_files/prepare_storage.sh 
```

### 5. Run the script

```
python active_learning/run.py
```

If no error pops up, the DSVM is ready for now.

### Final Step

Go to the address of you web app, and start playing with it.