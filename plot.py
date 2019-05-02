import os
import torch
import pyro
import sys

import utils

import matplotlib
matplotlib.use('pdf')
#matplotlib.use('svg')
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

plt.style.use(['seaborn-white', 'seaborn-paper', 'seaborn-ticks'])
#matplotlib.rc('font', family='Latin Modern Roman')
#mathtext.fontset
plt.rc('text', usetex=False)
plt.rc('font', family='serif')

viridis = plt.get_cmap('viridis').colors
viridis = [viridis[i] for i in [100, 240, 150, 0]]
#tab20b = plt.get_cmap('tab20b').colors
#tab20b = plt.get_cmap('winter').colors
colors = [None for _ in range(4)]
colors[0] = viridis[0]
colors[1] = plt.get_cmap('Paired').colors[3]
colors[2] = (0.75, 0., 0.) #tab20b[13]
colors[3] = (0.3, 0.3, 0.3)

def cm2inch(value):
  return value/2.54

def metrics(results, exp_name, model_name, ylim=None, ylim_ess=None):  #, ylim=(-3., 3.)):
    dpi = 200
    fig = plt.figure(figsize=(1200/dpi, 600/dpi))
    fig.suptitle(f'{model_name.replace("_","-")}') #, y=1.05)

    if ylim == None:
        ax = fig.add_subplot(1, 2, 1, title=f'Bounds', xlabel=r'epochs', ylabel=r'nats')
    else:
        ax = fig.add_subplot(1, 2, 1, title=f'Bounds', xlabel=r'epochs', ylabel=r'nats', ylim=ylim)

    epochs = results[0]['epoch']
    elbo = torch.stack([torch.tensor(r['elbo']) for r in results])
    iwae_10 = torch.stack([torch.tensor(r['iwae_10']) for r in results])
    iwae_100 = torch.stack([torch.tensor(r['iwae_100']) for r in results])
    iwae_5000 = torch.stack([torch.tensor(r['iwae_5000']) for r in results])
    ess = torch.stack([torch.tensor(r['ess']) for r in results])

    elbo_mean = elbo.mean(dim=0).cpu().numpy()
    iwae_10_mean = iwae_10.mean(dim=0).cpu().numpy()
    iwae_100_mean = iwae_100.mean(dim=0).cpu().numpy()
    iwae_5000_mean = iwae_5000.mean(dim=0).cpu().numpy()
    ess_mean = ess.mean(dim=0).cpu().numpy()

    elbo_sd = elbo.std(dim=0).cpu().numpy()
    iwae_10_sd = iwae_10.std(dim=0).cpu().numpy()
    iwae_100_sd = iwae_100.std(dim=0).cpu().numpy()
    iwae_5000_sd = iwae_5000.std(dim=0).cpu().numpy()
    ess_sd = ess.std(dim=0).cpu().numpy()

    #for idx, (guide_fns, exp_prefix) in enumerate(zip(guide_fns, exp_prefixes)):
    ax.fill_between(epochs, elbo_mean-elbo_sd, elbo_mean+elbo_sd, color=colors[2], alpha=0.25)
    ax.fill_between(epochs, iwae_10_mean-iwae_10_sd, iwae_10_mean+iwae_10_sd, color=colors[0], alpha=0.25)
    ax.fill_between(epochs, iwae_100_mean-iwae_100_sd, iwae_100_mean+iwae_100_sd, color=colors[1], alpha=0.25)
    ax.fill_between(epochs, iwae_5000_mean-iwae_5000_sd, iwae_5000_mean+iwae_5000_sd, color='black', alpha=0.25)

    ax.plot(epochs, elbo_mean, color=colors[2], label='elbo')
    ax.plot(epochs, iwae_10_mean, color=colors[0], label='iwae-10')
    ax.plot(epochs, iwae_100_mean, color=colors[1], label='iwae-100')
    ax.plot(epochs, iwae_5000_mean, color='black', label='iwae-5000')

    ax.legend()
    ax.xaxis.set_tick_params(width=0.5)
    ax.yaxis.set_tick_params(width=0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    sns.despine()

    if ylim_ess == None:
        ax = fig.add_subplot(1, 2, 2, title=f'ESS', xlabel=r'epochs', ylabel=r'samples')
    else:
        ax = fig.add_subplot(1, 2, 2, title=f'ESS', xlabel=r'epochs', ylabel=r'samples', ylim=ylim_ess)

    #for idx, (guide_fns, exp_prefix) in enumerate(zip(guide_fns, exp_prefixes)):

    ax.fill_between(epochs, ess_mean-ess_sd, ess_mean+ess_sd, color=colors[2], alpha=0.25)
    ax.plot(epochs, ess_mean, color=colors[2])

    ax.xaxis.set_tick_params(width=0.5)
    ax.yaxis.set_tick_params(width=0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    sns.despine()

    #plt.subplots_adjust(top=1.1)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    fig.savefig(f'./results/{exp_name}_metrics.pdf')
    plt.close(fig)
