import os
import argparse

import numpy
import pyro_models
from pyro_models.utils import json_file_to_mem_format
from utils import save, load
from model_constants import model_constants

import pyro.poutine as poutine
from pyro.infer.mcmc import MCMC, NUTS
from pyro.infer.abstract_infer import TracePredictive

import torch

import pystan

def pyro_nuts(name):
    models = pyro_models.load()
    md  = models[name]
    data = pyro_models.data(md)
    model = md['model']

    # will need to block out discrete vars for cjs models
    # NOTE: dont jit with minibatches
    nuts_kernel = NUTS(model, jit_compile=True, max_tree_depth=5)
    posterior = MCMC(nuts_kernel, num_samples=2000, warmup_steps=2000, disable_progbar=True)
    file_name = '{}_nuts_pyro.pkl'.format(name)
    if os.path.exists(file_name):
        print('Using cached model {}'.format(file_name))
        mcmc_md = load(file_name)
        for i, k in mcmc_md.items():
            setattr(posterior, i, k)
    else:
        posterior = posterior.run(data, {})
        mcmc_md = {'exec_traces': posterior.exec_traces,
                   'log_weights': posterior.log_weights,
                   'chain_ids': posterior.chain_ids,
                   '_idx_by_chain': posterior._idx_by_chain,
                   '_categorical': posterior._categorical
                   }
        print('Saving mcmc traces to {}'.format(file_name))
        save(mcmc_md, file_name)

    obs = model_constants[name]['sitename'] if 'sitename' in model_constants[name] else 'y'
    # set obs to None to sample from the posterior predictive
    true_samples = data[obs]
    data[obs] = None
    mses = []
    for i in range(10):
        posterior_pred = TracePredictive(model, posterior, num_samples=1000).run(data, {})
        marginal = posterior_pred.marginal([obs])
        pred_samples = marginal.support(flatten=True)[obs]
        mse = (pred_samples - true_samples).pow(2).mean()
        print(mse)
        mses.append(mse)
    return mses


def stan_nuts(file_name, data, src_code, n_samples=2000, n_chains=10):
    """
    Runs the No U-Turn Sampler using Pystan

    :param data: model data
    :param src_code: Stan source code as a string
    :param init_values: initial values for NUTS initialization
    :param n_samples: number of samples for NUTS
    :returns: OrderedDict of variables keyed on variable name
    """
    if os.path.exists(file_name):
        print('Using cached model {}'.format(file_name))
        model = load(file_name)
    else:
        model = pystan.StanModel(model_code=src_code)
        print('Caching compiled stan model to: {}'.format(file_name))
        save(model, file_name)
    fit = model.sampling(data=data, iter=n_samples, warmup=1000, chains=n_chains, algorithm="NUTS")
    print(fit.extract())

    print('saving summary object to: {}'.format(file_name[:-3] + 'out'))
    save(fit.summary(), file_name[:-3] + 'out')
    return fit.summary()


def main(args):
    book, model_name = args.name.split('.')
    data_path = pyro_models.__path__[0]+'/' + book + '/' + model_name + '.py.json'
    src_path = 'stan_models/' + model_name + '.stan'
    file_name = book + '.' + model_name + '.pkl'
    data = json_file_to_mem_format(data_path)
    if args.backend == 'pyro':
        mses = pyro_nuts(args.name)
        m, std = torch.tensor(mses).mean(), torch.tensor(mses).std()
        save([m.item(), std.item()], args.results_dir + args.prefix + '_nuts_' + file_name)
        print(model_name)
        print([m.item(), std.item()])
        return mses
    else:
        with open(src_path, "r") as f:
            src_code = f.read()
        return stan_nuts(file_name, data, src_code)


if __name__ == "__main__":
    # assumes all pyro models and json data are in pyro_models/
    # assumes all stan models are in stan_models
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--name', type=str, default='arm.wells_dist', help="name of model")
    parser.add_argument('-r', '--results-dir', type=str, default='./', help="directory in which to save results")
    parser.add_argument('-n', '--num-samples', type=int, default=2000, help="num samples")
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--backend', type=str, default="pyro", help="{pyro, stan}")
    args = parser.parse_args()

    mses = main(args)
