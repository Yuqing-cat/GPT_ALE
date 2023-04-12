import json
from azure.ai.ml.entities import Environment, BuildContext, Workspace
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient


def main(config):
    """
    create the aml environment
    """
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
        credential,
        config['aml']['subscription_id'],
        config['aml']['resource_group'],
        config['aml']['workspace']
    )

    build = BuildContext(
        dockerfile_path="Dockerfile",
        path="./aml/docker_context"
    )

    environment = Environment(
        name="palantir",
        build=build
        )

    ml_client.environments.create_or_update(environment)

if __name__ == "__main__":
    # read configuration file
    with open("configs/dbpedia_14/defaults/aml.json", "r") as f:
        config = json.load(f)

    main(config)
