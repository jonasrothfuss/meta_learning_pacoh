[![Build Status](https://travis-ci.com/jonasrothfuss/meta_learning_pacoh.svg?branch=master)](https://travis-ci.com/jonasrothfuss/meta_learning_pacoh)

# PACOH: Bayes-Optimal Meta-Learning with PAC-Guarantees
This repository provides ssource code corresponding the paper [*PACOH: Bayes-Optimal Meta-Learning with PAC-Guarantees*](https://arxiv.org/abs/2002.05551). 
In particular, the **meta_learn** package holds implementations of the following meta-learning algorithms:

* PACOH-MAP
* PACOH-VI
* PACOH-SVGD

Additionaly, an implementation of MAML as well as Neural Processes (NPs), based on third party code is comprised 
in the the **meta_learn** package.

The **experiments** directory holds code for synthetic task-environments and provides the necessary scripts to reproduce 
the experimental results reported in the paper.

## Installation
To install the minimal dependencies needed to use the meta-learning algorithms, run in the main directory of this repository
```bash
pip install .
``` 

For full support of all scripts in the repository, for instance to reproduce the experiments, further dependencies need to be installed. 
To do so, please run in the main directory of this repository 
```bash
pip install -r requirements.txt
``` 


## Usage
The following code snippet demonstrates the core functionality of the meta-learners provided in this repository. 
In addition, we refer to **demo.py** and **demo.ipynb** for a code example.

```python
""" A) generate meta-training and meta-testing data """
from experiments.data_sim import SinusoidDataset
task_environment = SinusoidDataset()
meta_train_data = task_environment.generate_meta_train_data(n_tasks=20, n_samples=5)
meta_test_data = task_environment.generate_meta_test_data(n_tasks=20, n_samples_context=5, n_samples_test=50)


""" B) Meta-Learning with PACOH-MAP """
from meta_learn import GPRegressionMetaLearned
meta_gp = GPRegressionMetaLearned(meta_train_data, weight_decay=0.2)
meta_gp.meta_fit(meta_test_data, log_period=1000)


"""  C) Meta-Testing with PACOH-MAP """
x_context, y_context, x_test, y_test = meta_test_data[0]

# target training in (x_ontext, y_context) & predictions for x_test
pred_mean, pred_std = meta_gp.predict(x_context, y_context, x_test)

# confidence intervals predictions in x_test 
ucb, lcb = meta_gp.confidence_intervals(x_context, y_context, x_test, confidence=0.9)

# compute evaluation metrics on one target task
log_likelihood, rmse, calib_error = meta_gp.eval(x_context, y_context, x_test, y_test)

# compute evaluation metrics for multiple tasks / test datasets
log_likelihood, rmse, calib_error = meta_gp.eval_datasets(meta_test_data)
```


## Reproducing the experiments
Below we point to the experiment scripts that were used to generate the results reported in the paper. 
Note that all of the experiment scripts use multiprocessing and were written for machines / high-performance 
clusters designed heavy workloads. Please take this into consideration, before launching any of the experiment scripts.

### Meta-overfitting experiments

To run the experiments:

```bash
python experiments/meta_overfitting_v2/meta-overfitting-pacoh-map.py
python experiments/meta_overfitting_v2/meta-overfitting-mll.py

``` 

To generate the plots in the paper:

```bash
python experiments/meta_overfitting_v2/plots_meta_overfitting_v2_map_vs_mll_paper.py
``` 

### Hyper-Parameter Search for PACOH and MLAP
The following command will launch multiple hyper-parameter tuning session with 
[ray tune](https://ray.readthedocs.io/en/latest/tune.html), based on [hyperopt](http://hyperopt.github.io/hyperopt/).
```bash
python experiments/hyperparam_search/launch_hyperparam_sweeps.py
```

### Reproducing the baselines

```bash
python experiments/baselines/baseline_comparison.py
```
