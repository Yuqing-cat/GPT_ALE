# Guidance to Provision Azure Resource for GPT-ALE

## Prerequisite
The provided Azure Resource Manager (ARM) template deploys the following resources, please make sure you have enough quota in the subscription and region you are deploying this in. You can view your quota and make request on Azure portal.

You need to have quota to provision the following resources:

1. An Azure Data Science Virtual Machine (Standard_D4s_v3, might scale up in future use)
1. An Azure Storage Account
1. Azure App Service and corresponding App Service plan


## Provision Azure Resources using ARM Template
1. Create an Azure Resource Group in the [Azure Portal](https://portal.azure.com).
1. Navigate to your Resource Group and click the **Create** Button, and select **template deployment (deploy using custom templates)** from Marketplace.
1. Click "Build your own template in the Editor", then paste the content of [this file](./setup_files/azure_provision.json) into the editor, then click "Save".
1. In "Basics" Tab, check all the resource names that is going to be used. You may have to change some names, if they are already taken by other users.
1. Add an Admin Password (and remember it)
1. Click “Review+Create”, then wait for the deployment to complete.

 
## Post-deployment configuration

### WebApp configuration

The WebApp is configured to use a docker image from the Azure Container Registry (ACR). You can either build your own docker image and push it to your on ACR, or you can contact us for the password for our ACR.

To configure the WebApp to pull the docker image for gpt-ale, you need to go to `Settings -> Configuration -> Applications Settings` and review these settings:

| Key      | Value | Comment     |
| :---        |    :----:   |          ---: |
| DOCKER_REGISTRY_SERVER_URL        | palantirdemoacr.azurecr.io       | Replace w/ the name of your ACR, if you are not using ours |
| DOCKER_REGISTRY_SERVER_USERNAME   | palantirdemoacr                  | Replace w/ your username, if you are not using our ACR |
| DOCKER_REGISTRY_SERVER_PASSWORD   | intentionally_left_blank         | **REDACTED**, please contact GPT-ALE Project Team |

Alternative, you can configure the Web App to have ACR imagepull permission. [Yuqing to add more]

### Connection String for Storage Account

The connection string of the storage account need also be added to the WebApp's Configuration Section, with Key "CONNECTION_STR". This should already be set correctly, but it may be worth reviewing it.

### Preparing Storage Blob Storage Container

Get into the storage account, and create a new blob container named "demo".

