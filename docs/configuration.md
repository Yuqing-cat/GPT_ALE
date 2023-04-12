# Configuration

## Folder structure of configuration files

We recommend to separate configurations by dataset. We provide one example configuration file for the HuggingFace dataset [dbpedia_14](https://huggingface.co/datasets/dbpedia_14).

These configuration can be found at [configs/dbpedia_14/](configs/dbpedia_14/).

This dictory contains several folders. First, a folder `default`, which contains your default settings for this dataset.

Then there are various other folders, that define specific variations of the default settings. For simplicity, these configurations only need to contain those settings that differ from the defaults.

## Configuration Files and Sections

A project configuration consists the following configuration files to be present.

- `configs/dbpedia_14/defaults/aml.json` - settings for the AzureML workspace (optional, only required for benchmarking on [AzureML](https://ml.azure.com))
- `configs/dbpedia_14/defaults/blob_storage.json` - settings for blob storage (optional, not required when performing benchmarking locally, i.e. w/o UI and storing data locally)
- `configs/dbpedia_14/defaults/config.json` - main configuration file, see sections below
- `configs/dbpedia_14/defaults/openai.json` - configuration for Azure OpenAI API (not required if your dataset contains ground truth labels and you don't what to use GPT-3.5 at all)
- `configs/dbpedia_14/defaults/samplers.json` - configuration for the various samples used for active learning
- `configs/dbpedia_14/defaults/sme.json` - configuration of a simulated human annotator (see, [below](configuratin.md#sme))

For the configuration for Azure Blob Storage ([documentation](https://learn.microsoft.com/en-us/azure/storage/blobs/sas-service-create-dotnet?tabs=dotnet)), Azure ML ([documentation](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment)), and Azure OpenAI API ([documentation](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/overview)), please refer to the product documentation.

Below, we provide more details on the various documentation settings.

## Data (config.json)

These are settings for your dataset, such as the name of your dataset, which columns contains the inputs and targets, the name of splits (e.g. "train" and "test").

## Model (config.json)

This section defined which embedding model you want to use (e.g. bert-base-uncased), hyperparameters for training the student model.

## Samplers (samplers.json)

This file contains the settings for the various samplers that can be used for active learning.

## Misc (config.json)

Finally, there are miscellaneous settings, such as whether you want to use GPT-3 for label generation. 

