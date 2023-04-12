# GPT - ALE (Generative Pre-trained Transformer - Active Learning Engine)


> **DISCLAIMERS**: 
> 1. This is a research project and is not intended for production use. In particular, even though this project uses state of the art approaches to validate and calibrate ML models, there is no guarantee that these models will perform as expected on out-of-sample data.
> 1. Using this project requires the provisioning of an [Azure OpenAI Service](https://azure.microsoft.com/en-us/products/cognitive-services/openai-service/) endpoint. This service has a [limited access policy](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/limited-access), to promote responsible use and limit the impact of high-risk use cases. You may be required to provide additional information to gain access to this service. Please also review our [responsible AI guidelines](https://learn.microsoft.com/en-us/azure/cognitive-services/responsible-use-of-ai-overview).

Data labeling and categorization is a critical step in the machine learning lifecycle. However, it is often a time-consuming and expensive process.

GPT-ALE reduces the time and cost of data labeling, by combining state-of-the-art foundation models (e.g. GPT), active learning, and regularized self-training. It uses active learning strategies to identify samples that help it simultaneously understand the user's intent and the dataset, and offers suggestions to the user as to how to label the data.

Once the user provides a reasonable amount of labeled samples, GPT-ALE uses regularized self-training and knowledge distillation to create a student model. GPT-ALE uses few-shot learning with dynamic prompt fine-tuning as it progresses in understanding the data and user intent. The student model is validated and calibrated using the ground truth labels provided by the user, which may occur over several iterations.

When the user is satisfied with the performance of the student model, it can be deployed to index a large dataset at a very high throughput. This allows the user to efficiently and accurately process and analyze large amounts of data.

## Documentation Overview

This documentation is divided into the following sections:

- [Provision Azure Resources](docs/provision_azure_resources.md) - This describes how to provision the Azure resources needed to run GPT-ALE.
- [Setup DSVM](docs/setup.md) - This describes how to setup the DSVM to run GPT-ALE.
- [ALE Configuration](docs/configuration.md) - This describes the main hyperparameter choices for GPT-ALE
- [Benchmarking](docs/benchmarking.md) - If you have an annotated dataset, you can use this to benchmark your configuration this dataset, without human interaction.
- [Front-End Guidance](docs/frontend_guidance.md) - Documentation for the front-end application that interacts with GPT-ALE.

## Troubleshooting

Please file an [issue](https://github.com/Azure/GPT_ALE/issues) if you encounter one.

You may also consult the [project wiki](https://github.com/Azure/GPT_ALE/wiki)


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.


## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
