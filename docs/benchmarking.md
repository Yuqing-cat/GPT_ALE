# Benchmarking with simulated SME

We provide a script, which can be used to a human annotator. Briefly, the simulated SME responds to the annotation requests from the active learning engine, instead of the a user working with the UI.


## Pre-requisites

1. Clone this repository: `git clone https://github.com/Azure/GPT_ALE.git`
1. Change directory: `cd GPT_ALE`
1. Set up the conda environment: `conda env create -f environment.yml`

## Getting-Started <a id="getting-started"></a>

> Note: We will use the [dbpedia_14](https://huggingface.co/datasets/dbpedia_14) dataset for this getting started

These are the basic steps for getting started:

1. Download the dataset: `python scripts/download_huggingface_dataset.py --config_path configs/dbpedia_14/0`
1. (recommended) Downsample the dataset, to speed things up: `python active_learning/subset_df.py --config_path configs/dbpedia_14/0`
1. Data pre-processing (including sentence embedding generation): `python active_learning/data_prep.py --config_path configs/dbpedia_14/0`
1. Start the active learning engine: `python active_learning/run.py --config_path configs/dbpedia_14/0`
1. (optional) Start MLFlow UI to monitor progress: `mlflow ui`
1. Start the simulated SME: `python active_learning/sme.py --config_path configs/dbpedia_14/0`

## Configuration of simulated SME <a id="sme"></a>

### label/target mapping
The simulated SME is a script that simulates a human annotator. It can be used for benchmarking.

The main configuration optios are:

- label_dict: the labels that the SME is going to assign to different ground truth categories. For example, if your dataset includes the label "Politician" and "Athlete", with a target values of 4 and 10, respectively, you would define the label_dict as follows:

```json

label_dict: {
    "Politician": 4,
    "Athlete": 10
}
```

This will result in a Politician being assigned a target value of 0, and Athelte a target value of 1. All other categories will be labeled as Other/2.

### Label Budget (`budget`)

This setting determines how many samples the SME is willing to annotate per setting.

### Mapping File (`mapping_file`)
This determines where the label/target mapping is stored. This file will be written by run.py after startup. It is important that the file is up-to-date when you start the SME, otherwise the SME will not be able to map the labels to the target values.

### Error Rate (`error_rate`)

How often the SME makes a mistake. This is a value between 0 and 1. The SME will randomly assign a label to a sample, with the probability of the error rate. For example, if the error rate is 0.1, the SME will make a mistake in 10% of the cases.

## Start the SME

The simulated SME can be started like so: `python  active_learning/sme.py --config_path configs/<dataset>/<configuration>`


## Benchmarking using Azure ML

Instead of performing benchmarking locally, you can also perform benchmarking on Azure ML. This allows you to test various dataset configurations in parallel.

First, create the configuration files that define the different test cases.

Then execute the script `aml/submit_aml.py` providing the configuration files as input arguments to the script, e.g.:

`python aml/submit_aml.py --configs configs/<dataset>/<config> configs/<dataset>/<config> configs/<dataset>/<config>`

alternatively, you can have the script submit all configurations in a directory, e.g.:

`python aml/submit_aml.py --config_path configs/<dataset>`