import os
import uuid
import glob
import json
import argparse

from azure.identity import DefaultAzureCredential

# Import the client object from the SDK library
from azure.storage.blob import BlobClient, BlobServiceClient, ContainerClient

    
def load_config(config_path):
    config_files = glob.glob(os.path.join(config_path, "*.json"))

    if len(config_files) == 0:
        raise Exception("No config files found in {}".format(config_path))

    config = {}
    for cf in config_files:
        with open(cf, "r") as f:
            config.update(json.load(f))
        
    return config

def upload(container_client, filename):
    # Open a local file and upload its contents to Blob Storage
    with open(filename, "rb") as data:
        container_client.upload_blob(data=data, name=filename, overwrite=True)
        print(f"Uploaded {filename} to {container_client.url}")

def main():    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_path', type=str, default='configs/dbpedia_14/defaults')
    argparser.add_argument("--config", nargs="+", help="key-value pair for configuration settings")
    args = argparser.parse_args()

    config = load_config(args.config_path)

    credential = DefaultAzureCredential()

    # Instantiate a ContainerClient
    connection_string = config['blob_storage']['connection_string']
    container_name = config['blob_storage']['container_name']
    container_client = ContainerClient.from_connection_string(connection_string, container_name)
    
    # create filename and upload
    dataset = config['data']['dataset']
    subset = config['data']["subset"]
    for split in ["train", "test"]:
        if subset is None:
            filename = os.path.join(os.path.join(config['data']['data_path'], dataset, "data_" + split + "_proc.csv"))
        else:
            filename = os.path.join(os.path.join(config['data']['data_path'], dataset + "_" + subset, "data_" + split + "_proc.csv"))
        upload(container_client, filename)

    
if __name__ == "__main__":
    main()