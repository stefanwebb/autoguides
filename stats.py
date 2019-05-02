from __future__ import absolute_import, division, print_function

import math
import warnings
import sys

import torch

from pyro.distributions.util import is_identically_zero
from pyro.infer.elbo import ELBO
from pyro.infer.enum import get_importance_trace
from pyro.infer.util import is_validation_enabled, torch_item
from pyro.util import check_if_enumerated, warn_if_nan


class Stats(ELBO):
    def __init__(self,
                 population_effects={},
                 num_particles=1000,
                 max_plate_nesting=float('inf'),
                 max_iarange_nesting=None,  # DEPRECATED
                 strict_enumeration_warning=True):
        if max_iarange_nesting is not None:
            warnings.warn("max_iarange_nesting is deprecated; use max_plate_nesting instead",
                          DeprecationWarning)
            max_plate_nesting = max_iarange_nesting
        self.population_effects = population_effects

        super(Stats, self).__init__(num_particles=num_particles,
                                        max_plate_nesting=max_plate_nesting,
                                        vectorize_particles=True,
                                        strict_enumeration_warning=strict_enumeration_warning)

    def _get_trace(self, model, guide, *args, **kwargs):
        """
        Returns a single trace from the guide, and the model that is run
        against it.
        """
        model_trace, guide_trace = get_importance_trace(
            "flat", self.max_plate_nesting, model, guide, *args, **kwargs)
        if is_validation_enabled():
            check_if_enumerated(guide_trace)
        return model_trace, guide_trace

    def loss(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Evaluates the ELBO with an estimator that uses num_particles many samples/particles.
        """

        # grab a vectorized trace from the generator
        population_effects = {}
        random_effects = {}
        for _, guide_trace in self._get_traces(model, guide, *args, **kwargs):
            for k,v in guide_trace.nodes.items():
                if v['type'] == 'sample' and not k.startswith('_'):
                    val = v['value']
                    mean = val.mean(dim=0).cpu().detach().numpy()
                    std = val.std(dim=0).cpu().detach().numpy()
                    N = val.size(0)
                    sorted_val, _ = torch.sort(val, dim=0)
                    top_quantile = sorted_val[int(round(float(N)*0.75))-1].cpu().detach().numpy()
                    bottom_quantile = sorted_val[int(round(float(N)*0.25))-1].cpu().detach().numpy()

                    if k in self.population_effects:
                        population_effects[k] = {'mean':mean, 'std':std, 'CI':(bottom_quantile,top_quantile)}
                    else:
                        random_effects[k] = {'mean':mean, 'std':std, 'CI':(bottom_quantile,top_quantile)}

        stats_params = {'population_effects':population_effects, 'random_effects':random_effects}
        return stats_params
