# Deep Gaussian Processes with Power EP

Alpha-divergence Minimization for Deep Gaussian Processes

Requirements: Python3.6, tensorflow1.15.0, tensorflow_probability=0.8, numpy1.19.4, psutil5.6.0

You can install all the Python dependencies needed to execute the experiments with pip by running:

``` bash
pip install -r requirements.txt

```

## Experiments

This repository comes with a series of experiments to evaluate the performance of the proposed approach:
 * Bimodal and heteroscedastic problems.
 * UCI regression
 * UCI binary classification
 * UCI multiclass classification
 * Big datasets: Year (regression), Airlines delays (regression and binary classification), HIGGS (binary classification), MNIST (multiclass classification)

### Experimental data 

In order to run the experiments, the first step is to uncompress the data and the splits in the datasets folder. For the UCI data, uncompress the corresponding file in .tgz format inside the datasets folder (regression.tgz, binary.tgz, multiclass.tgz). 


### Running the experiments

There are three subfolders in the experiments directory, one of each corresponding to a different type of experiment:
 * predictive_bimodal: The bimodal and heteroscedastic problems experiments.

```bash
Usage:
	python3 multimodal.py <alpha> (1|2)
```

 * uci: All the uci datasets.

```bash
Usage:
	python3 run_uci.py <dataset_name> <split> <n_layers> <alpha>
```

 * performance_time: The big datasets.

```bash
Usage:
	python3 run_exp.py <dataset_name> <n_layers> <alpha>
```
