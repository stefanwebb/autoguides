import os
import sys
import time
import argparse
import pickle
import importlib.util
import warnings
import logging
from datetime import datetime

import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, RenyiELBO, TraceEnum_ELBO, TraceTailAdaptive_ELBO
from pyro.contrib.autoguide import AutoDelta, AutoDiagonalNormal, AutoMultivariateNormal, AutoIAFNormal, AutoDSFNormal, AutoDEFNormal
from pyro.poutine import trace
import pyro.optim as optim
import pyro_models
import pyro.poutine as poutine

import utils
from model_constants import model_constants
from ess import ESS
from mse import MSE
from stats import Stats


logging.basicConfig(filename='', filemode='a+', level=logging.WARNING)

guide_types = [AutoDiagonalNormal, AutoMultivariateNormal, AutoIAFNormal, AutoDSFNormal, AutoDEFNormal]
guide_map = {}
for g in guide_types:
    guide_map[g.__name__] = g


def run_experiment(args):
    models = pyro_models.load()
    model = models[args.model_name]
    data = pyro_models.data(model)
    N_data = data[model_constants[args.model_name]['dataname']].size(0) if 'dataname' in model_constants[args.model_name] else data['y'].size(0)

    # Convert string for guide to class
    guide_type = guide_map[args.guide_type]

    # Make unique experiment name and set directory
    exp_dir = args.results_dir+args.model_name+'/'
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)

    # guide_type, learning_rate, elbo_particles, tail_adaptive, decimate_lr, seed
    exp_name =  'g={}_lr={}_iw={}_tailadaptive={}_decimatelr={}_seed={}'.format(args.guide_type, args.learning_rate, args.elbo_particles, args.tail_adaptive, args.decimate_lr, args.seed) + (('_' + args.unique_prefix) if args.unique_prefix != '' else '')

    file_path = os.path.join(exp_dir, '{}.results'.format(exp_name))

    # Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialize Pyro
    pyro.enable_validation(True)
    pyro.clear_param_store()

    # Define model/data/guide/loss
    model_fn = model['model']
    data = (pyro_models.data(model), {})
    guide_fn = guide_type(model_fn)
    guide_fn(*data)

    sitename = model_constants[args.model_name]['sitename'] if 'sitename' in model_constants[args.model_name] else 'y'
    dataname = model_constants[args.model_name]['dataname'] if 'dataname' in model_constants[args.model_name] else 'y'
    stats = Stats(population_effects=model_constants[args.model_name]['population_effects'], num_particles=1000)

    loss = Trace_ELBO(vectorize_particles=True, num_particles=args.elbo_particles)

    # only calculate ess and mse for stan models
    ess = ESS(vectorize_particles=True, num_inner=1000, num_outer=10)
    mse = MSE(observed_sitename=sitename, observed_dataname=dataname, vectorize_particles=True, num_particles=1000)
    tail_elbo = TraceTailAdaptive_ELBO(vectorize_particles=True, num_particles=args.elbo_particles)
    iwae10 = RenyiELBO(vectorize_particles=True, num_particles=10)
    iwae100 = RenyiELBO(vectorize_particles=True, num_particles=100)
    iwae5000 = RenyiELBO(vectorize_particles=True, num_particles=5000)

    results = {'args':args, 'epoch':[], 'elbo':[], 'ess':[], 'mse':[], 'iwae_10':[], 'iwae_100':[], 'iwae_5000':[], 'kl_q_p':[], 'time':[]}
    pickle.dump( results, open(file_path, 'wb' ), protocol=pickle.HIGHEST_PROTOCOL )

    # Initialize the param store
    elbo_val = loss.loss(model_fn, guide_fn, *data)
    if args.decimate_lr:
        optimizer = optim.MultiStepLR({'optimizer': torch.optim.Adam, 'optim_args': {'lr': args.learning_rate}, 'milestones':[200], 'gamma': 0.1})
    else:
        optimizer = optim.Adam({'lr': args.learning_rate})

    # Perform variational inference
    svi = SVI(model_fn, guide_fn, optimizer, loss=loss)
    if args.tail_adaptive:
        tail_svi = SVI(model_fn, guide_fn, optimizer, loss=tail_elbo)

    time_per_log = 0.

    for i in range(args.num_epochs+1):
        if i % args.log_frequency == 0:
            with torch.no_grad():
                mse_val = mse.loss(model_fn, guide_fn, *data)
                ess_val = ess.loss(model_fn, guide_fn, *data)
                iwae_10_val = iwae10.loss(model_fn, guide_fn,*data)
                iwae_100_val = iwae100.loss(model_fn, guide_fn, *data)
                iwae_5000_val = iwae5000.loss(model_fn, guide_fn, *data)

        start = time.time()
        with poutine.scale(scale=1.0/N_data):
            if not args.tail_adaptive:
                # NOTE: loss_val is the loss *before* taking gradient step!
                loss_val = svi.step(*data) * N_data
            else:
                loss_val = loss.loss(model_fn, guide_fn, *data) * N_data
                tail_svi.step(*data)
        end = time.time()
        time_per_log += (end - start)

        if i % args.log_frequency == 0:
            results['epoch'].append(i)
            results['elbo'].append(loss_val)
            results['ess'].append(ess_val)
            results['mse'].append(mse_val)
            results['iwae_10'].append(iwae_10_val)
            results['iwae_100'].append(iwae_100_val)
            results['iwae_5000'].append(iwae_5000_val)
            results['kl_q_p'].append(loss_val - iwae_5000_val)
            results['time'].append(time_per_log)

            if i == args.num_epochs:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    results['stats'] = stats.loss(model_fn, guide_fn, *data)

            print('epoch {}'.format(i), 'elbo', loss_val, 'iwae-10/100/5000', iwae_10_val, iwae_100_val, iwae_5000_val, 'ess', ess_val, 'mse', mse_val, 'time', time_per_log)
            pickle.dump( results, open(file_path, 'wb' ), protocol=pickle.HIGHEST_PROTOCOL )
            time_per_log = 0.


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=int, default=0, help="cuda device")
    parser.add_argument('-r', '--results-dir', type=str, default='results/', help="directory in which to save results")
    parser.add_argument('-n', '--num-epochs', type=int, default=1000, help="num epochs per run")
    parser.add_argument('-m', '--model-name', type=str, default='arm.wells_dist', help="model name for model zoo")
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.1, help="learning rate for ADAM")
    parser.add_argument('-p', '--unique-prefix', type=str, default='', help="unique prefix to results file")
    parser.add_argument('-g', '--guide-type', type=str, default='AutoDiagonalNormal', help="guide type")
    parser.add_argument('-e', '--elbo-particles', type=int, default=10, help="number of IW samples in MC ELBO estimate")
    parser.add_argument('-t', '--tail-adaptive', dest='feature', action='store_true', help="whether to use tail-adaptive f-divergence loss")
    parser.add_argument('-dl', '--decimate-lr', dest='feature', action='store_true', help="whether to decimate the learning rate at 200 epochs")
    parser.add_argument('-lf', '--log-frequency', type=int, default=20, help="frequency at which to log output")
    parser.add_argument('-s', '--seed', type=int, default=0, help="random seed to use")
    parser.add_argument('--test', default=False, action='store_true', help="dry run")

    parser.set_defaults(tail_adaptive=False)
    parser.set_defaults(decimate_lr=False)
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        print('using cuda on device: ' + str(args.device))
        with torch.cuda.device(args.device):
            run_experiment(args)
