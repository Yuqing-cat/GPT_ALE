# Benchmarking on Azure ML

The files in this directory can be used to do benchmarking and unit testing on Azure ML, taking advantage of scalable compute clusters for parallel execution of individual benchmark settings or unit tests. 

At least for now, the basic setting is that we execute two scripts `run.py` and `sme.py` on the same compute node.

- `run.py` trains uses active learning to train a student model.
- `sme.py` simulates a human subject matter expert who annotates the data.

## Workflow

1. Create an [Azure ML](https://ml.azure.com) workspace.
1. Create `configs/defaults/aml.json` (based on `configs/templates/aml.json`)
1. Upload data `aml/upload_data.py`.
1. Create environment (docker container) for training: `aml/create_env.py`.
1. Submit jobs: `aml/submit_aml.py`
