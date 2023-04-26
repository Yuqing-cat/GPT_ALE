# Internal Bug Bash Guide

This guide is intended to help you get started with the internal bug bash.
- DSVM Setup
- Blob Storage Setup
- Web App Setup

## DSVM Setup
This section includes the steps to create a DSVM and install the required packages.
### Step 1. Create a DSVM
Click link [Data Science Virtual Machine - Ubuntu 20.04](https://ms.portal.azure.com/#create/microsoft-dsvm.ubuntu-20042004) to create a DSVM.
1. Click `Create` and fill in the required information. You can use the default values for most of the fields. Please note that GPU may required for some scenarios. You can resize the VM after the creation.
2. Click `Review + create` and `Create` to create the DSVM.

### Step 2. Install the Required Packages
1. Connect to the DSVM. You can pick a way through the `Connect` tab of the DSVM.
2. Click `Clone` and `Generate Git Credentials` and save your password in `https://dev.azure.com/msazure/One/_git/gpt_ale`
3. Clone the repo: `git clone https://msazure@dev.azure.com/msazure/One/_git/gpt_ale`
4. Enter the password when prompted.
5. Change directory: `cd gpt_ale`
6. Change branch to `wopauli/thebackend`: `git checkout wopauli/thebackend` which contains the latest changes.
7. Run `conda env create -f environment.yml` to create the conda environment.
8. `conda activate gpt_ale`
9. Issues:
    - may need to install `rpdb` manually: `pip install rpdb`
    - may get OSError: `OSError: /anaconda/envs/gpt_ale/lib/python3.9/site-packages/nvidia/cublas/lib/libcublas.so.11: undefined symbol: cublasLtGetStatusString, version libcublasLt.so.11` 
      - `pip uninstall nvidia_cublas_cu11`

### Step 3. (Optional) Prepare the Data
This step is optional, you can copy files from `https://gpt_alestorageaccount.blob.core.windows.net/data/dbpedia_14_dbpedia_14` instead. 
#### Option A: Prepare your own data
1. Download the dataset: `python scripts/download_huggingface_dataset.py --config_path configs/dbpedia_14/0`
2. Data Pre-processing: `python active_learning/data_prep.py --config_path configs/dbpedia_14/0`

#### Option B: Use the provided pickle files
1. download files from `https://gpt_alestorageaccount.blob.core.windows.net/data/dbpedia_14_dbpedia_14` to `data/dbpedia_14/_dbpedia_14`
- `wget "https://gpt_alestorageaccount.blob.core.windows.net/data/dbpedia_14_dbpedia_14/data_test.pkl" -O data_test.pkl`
- add your SAS token to the end of the URL to avoid 404 Error

### Step 4. Start Running:
1. Start the active learning engine: `python active_learning/run.py --config_path configs/dbpedia_14/0`
2. Start the simulated SME: `python active_learning/sme.py --config_path configs/dbpedia_14/0`
3. Start MLFlow UI to monitor progress: `mlflow ui`


### Blob Storage Setup
Reference: 
- [sampleFileCacheConfig.yaml](https://github.com/Azure/azure-storage-fuse/blob/main/sampleFileCacheConfig.yaml)
- [What is BlobFuse?](https://learn.microsoft.com/en-us/azure/storage/blobs/blobfuse2-what-is)
1. Create a storage account, ideally in the same region as the DSVM
2. Create a container named "data"
3. Generate Shared Access Signature (SAS) token, with permissions `racwdl` (everything except immutable storage) and a reasonable expiry time (e.g. 3 months)
4. Generate configuration based on the following file
5. Create a file config.yaml in your home directory

```yml
allow-other: true
 
logging:
  type: syslog
  level: log_debug
 
components:
  - libfuse
  - file_cache
  - attr_cache
  - azstorage
 
libfuse:
  attribute-expiration-sec: 120
  entry-expiration-sec: 120
  negative-entry-expiration-sec: 240
 
file_cache:
  path: /mnt/tempcache
  timeout-sec: 120
  max-size-mb: 4096
 
attr_cache:
  timeout-sec: 7200
 
azstorage:
  type: block
  account-name: <name of storage account>
  endpoint: https://<name of storage account>.blob.core.windows.net
  mode: sas
  sas: <your sas>
  container: data

```

```sh
wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
sudo apt-get update
sudo apt-get install libfuse3-dev fuse3
sudo apt-get install blobfuse2
sudo mkdir /mnt/tempcache
mkdir ~/mycontainer
blobfuse2 mount mycontainer --config-file=./config.yaml
cd ~/gpt_ale
ln -s ../mycontainer data

```
Tips:
- Add `foreground: true` at top of your config and "fuse-trace: true" to your libfuse section in config.yaml to see more logs.


### Web App Setup
1. Create a web app with Managed Identity
2. Configure the Managed Identity to have access to the ACR
   - Assign the role `AcrPull` to the Managed Identity of App Service
3. Use the ACR image [`gpt_ale.azurecr.io/gpt_ale:latest`](https://ms.portal.azure.com/#view/Microsoft_Azure_ContainerRegistries/RepositoryBlade/id/%2Fsubscriptions%2Fa6c2a7cc-d67e-4a1a-b765-983f08c0423a%2FresourceGroups%2Fappsvc_linux_centralus_basic%2Fproviders%2FMicrosoft.ContainerRegistry%2Fregistries%2Fgpt_aledemoacr/repository/palntirapi) to deploy the web app
4. . Add the following environment variables to the web app in `Congiguration` -> `Application Settings`:
   - `"CONNECTION_STR":"<connection string of blob storage configured above>"`
   - remember to click `Save` to save the changes
5. Add redirect URL to AAD App Registration

TODO: 
1. Decouple AAD Client ID and AAD Tenant ID from the code
2. Make blob storage mount experience more smooth
3. Summarized the steps into automation scripts


### Feature Bug Bash Guide
[Bug Bash Feature List](https://microsoftapc-my.sharepoint.com/:w:/g/personal/yihgu_microsoft_com/EZ8U--_bdAdDmxbNiUqITHIBrtjN24OAMoA_IQKeXONkRw)