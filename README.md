# Improving Automated Variational Inference with Normalizing Flows 

Code for the paper, Webb, et al. *Improving Automated Variational Inference with Normalizing Flows*.

## Code

`run_models.py` - main test script to run VI on the models

`run_models.py` - test script for running NUTS on the models

`mse.py` - calculates the MSE for VI (used in `run_models`)

`stats.py` - calculates other stats of VI (used in `run_models`)

`ess.py` - calculates the ESS of VI (used in `run_models`)

`plot.py` - generates plots of learning curves

`model_constants.py` - contains metadata for the Pyro models. Used for visualization and metric calculations.

`stan_models/` - directory containing the Stan models

`pyro_models` - directory containing the Stan models written in Pyro

## Running
To run VI on the models, 
```
python run_models.py -n {num_epochs} -m {model name} -g {guide name} -lr {Adam lr} --elbo-particles {num_particles} --results-dir {output dir path}
```
eg:
```
python run_models.py -n 1000  -m arm.wells_dist -g AutoDiagonalNormal -lr 0.001  --elbo-particles 10  --results-dir results/ 
```

## Requirements
This project requires [Pytorch 1.0](https://pytorch.org/get-started/locally/) and custom code implemented in this [Pyro fork](https://github.com/stefanwebb/pyro/tree/for_jp).  If running the Stan backend for NUTS, it also requires [pystan](https://pypi.org/project/pystan/).
