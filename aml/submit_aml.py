# import required libraries
import json
from azure.ai.ml import MLClient
from azure.ai.ml import command
from azure.ai.ml import Input
from azure.identity import DefaultAzureCredential
from azure.ai.ml.constants import AssetTypes

import shutil
import os

def copy_required_files():
    """
    Copy required files to the aml/src folder
    """
    if os.path.exists("aml/src"):
        shutil.rmtree("aml/src")
    os.makedirs("aml/src", exist_ok=True)
    shutil.copytree("active_learning", "aml/src/active_learning")
    shutil.copytree("configs", "aml/src/configs")

def submit(config):
    """
    create and submit the aml job
    """
    # Enter details of your AML workspace
    subscription_id = config['aml']['subscription_id']
    resource_group = config['aml']["resource_group"]
    workspace = config['aml']["workspace"]

    credential = DefaultAzureCredential(
        exclude_cli_credential=False,
        exclude_environment_credential=True,
        exclude_managed_identity_credential=True,
        exclude_power_shell_credential=True,
        exclude_visual_studio_credential=True,
        exclude_visual_studio_code_credential=True,
        exclude_shared_token_cache_credential=True,
        exclude_interactive_browser_credential=True,
    )

    ml_client = MLClient(
        credential, subscription_id, resource_group, workspace
    )


    config_paths = ['configs/dbpedia_14/0', 'configs/dbpedia_14/1', 'configs/dbpedia_14/2', 'configs/dbpedia_14/3', 'configs/dbpedia_14/4']
    config_paths = ['configs/dbpedia_14/2', 'configs/dbpedia_14/4']

    for config_path in config_paths:
        inputs = {
            "data_path": Input(type=AssetTypes.URI_FOLDER, path="azureml://datastores/workspaceblobstore/paths/data/"),
            "config_path": config_path
        }

        job = command(
            code="./aml/src",  # local path where the code is stored
            command="""
                    python active_learning/sme.py --config_path ${{inputs.config_path}} --config data.data_path=${{inputs.data_path}} &
                    python active_learning/run.py --config_path ${{inputs.config_path}} --config data.data_path=${{inputs.data_path}} 
                    """,
            inputs=inputs,
            environment='palantir:2',
            compute="gpu-cluster-ssh",
            # display_name="palantir-benchmarking",
            description="Benchmarking run for Palantir",
            tags={"tag": inputs['config_path']},
        )

        # submit the command
        returned_job = ml_client.create_or_update(job)

    # config_paths = ['configs/dbpedia_14/0', 'configs/dbpedia_14/3']
    # for config_path in config_paths:
    #     inputs = {
    #         "data_path": Input(type=AssetTypes.URI_FOLDER, path="azureml://datastores/workspaceblobstore/paths/data/"),
    #         "config_path": config_path
    #     }
    #     for gpt3_error_rate in [0.0, 0.01, 0.05, 0.1, .15]:
    #         inputs['gpt3_error_rate'] = gpt3_error_rate
    #         job = command(
    #             code="./aml/src",  # local path where the code is stored
    #             command="""
    #                     python active_learning/sme.py --config_path ${{inputs.config_path}} --config data.data_path=${{inputs.data_path}} &
    #                     python active_learning/run.py --config_path ${{inputs.config_path}} --config data.data_path=${{inputs.data_path}} misc.gpt3_error_rate=${{inputs.gpt3_error_rate}} 
    #                     """,
    #             inputs=inputs,
    #             environment='palantir:1',
    #             compute="gpu-cluster",
    #             # display_name="palantir-benchmarking",
    #             description="Benchmarking run for Palantir",
    #             tags={"tag": inputs['config_path'], 'gpt3_error_rate': gpt3_error_rate},
    #         )

    #         # submit the command
    #         returned_job = ml_client.create_or_update(job)

    # config_path = 'configs/dbpedia_14/0'
    # inputs = {
    #     "data_path": Input(type=AssetTypes.URI_FOLDER, path="azureml://datastores/workspaceblobstore/paths/data/"),
    #     "config_path": config_path
    # }
    # for budget in [14, 28, 56, 112, 224]:
    #     inputs['budget'] = budget
    #     job = command(
    #         code="./aml/src",  # local path where the code is stored
    #         command="""
    #                 python active_learning/sme.py --config_path ${{inputs.config_path}} --config data.data_path=${{inputs.data_path}} sme.budget=${{inputs.budget}} &
    #                 python active_learning/run.py --config_path ${{inputs.config_path}} --config data.data_path=${{inputs.data_path}} 
    #                 """,
    #         inputs=inputs,
    #         environment='palantir:1',
    #         compute="gpu-cluster",
    #         # display_name="palantir-benchmarking",
    #         description="Benchmarking run for Palantir",
    #         tags={"tag": inputs['config_path'], 'budget': budget},
    #     )

    #     # submit the command
    #     returned_job = ml_client.create_or_update(job)

def main(config):
    copy_required_files()
    submit(config)

if __name__ == "__main__":
    # read configuration file
    with open("configs/dbpedia_14/defaults/aml.json", "r") as f:
        config = json.load(f)

    main(config)
