# Active Surrogate Estimators: An Active Learning Approach to Label-Efficient Model Evaluation

Hi, good to see you here! 👋

This is code for `Active Surrogate Estimators: An Active Learning Approach to Label-Efficient Model Evaluation'.


## Setup

We recommend you set up a conda environment like so:

```
conda-env update -f slurm/environment.yaml
conda activate ase
```


## Reproducing the Experiments

### Overview

* The `reproduce' folder contains scripts for running specific experiments.
* Execute a script as
```
sh reproduce/<script-name>.sh
```
* You can then create plots with the Jupyter Notebook at
```
notebooks/plots_paper.ipynb
```
* All scripts log continuously, so you should be able to create plots as the results are coming in.


### Experiments

* To recreate the distribution shift experiments, run the script `reproduce/Missing7.sh`.
    * To get results in reasonable time, we recommend starting this multiple times in parallel, e.g. across different compute nodes of a cluster. We ran it on ~30 GPUs for ~1 day, where each GPU ran the script three times in parallel (so 100 processes total). Different runs will automatically be combined by the evaluation script.
* To recreate the ResNet experiments, run the scripts `reproduce/ResNetCifar10.sh`, `reproduce/ResNetCifar100.sh`, and `reproduce/ResNetFMNIST.sh`.


## Details: Code Structure

* `main.py` is the main entry point into this code-base.
    * It executes a a total of  `n_runs` experiments for a fixed setup.
    * Each experiment:
        * Trains (or loads) one main model.
        * This model can then be evaluated with a variety of acquisition strategies.
        * Risk estimates are then computed for all acquisition strategies and all risk estimators.

* This repository uses `Hydra` to manage configs.
    * Look at `conf/config.yaml` or one of the experiments in `conf/...` for configs and hyperparameters.
    * Experiments are autologged and results saved to the `outputs/` directory.

* Different modules
    * `main.py` runs repeated experiments and orchestrates the whole shebang.
        * It iterates through all `n_runs` and `acquisition strategies`.
    * `experiment.py` handles a single experiment.
        * It combines the `model`, `dataset`, `acquisition strategy`, and `risk estimators`.
    * `datasets.py`, `aquisition.py`, `loss.py`, `risk_estimators.py`. Those should all contain more or less what you would expect.
    * `hoover.py` is a logging module.
    * `models/` contains all models, scikit-learn and pyTorch.
        * In `sk2torch.py` we have some code that wraps torch models in a way that lets them be used as scikit-learn models from the outside.

## And Finally

Thanks for stopping by!

If you find anything wrong with the code, please contact us.

We are happy to answer any questions related to the code and project.
