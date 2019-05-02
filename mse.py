from __future__ import absolute_import, division, print_function

import math
import sys
import warnings

import torch

import pyro
from pyro.distributions.util import is_identically_zero
from pyro.infer.elbo import ELBO
from pyro.infer.util import is_validation_enabled, torch_item
from pyro.util import check_if_enumerated, warn_if_nan
from pyro.poutine.util import prune_subsample_sites
from pyro.util import check_model_guide_match, check_site_shape, ignore_jit_warnings
from pyro import poutine

def get_importance_trace(graph_type, max_plate_nesting, model, guide, *args, count_traces=10, **kwargs):
    """
    Returns a single trace from the guide, and the model that is run
    against it.
    """
    guide_trace = poutine.trace(guide, graph_type=graph_type).get_trace(*args, **kwargs)

    with pyro.plate("data_plate", count_traces, dim=-(max_plate_nesting+1)):
        model_trace = poutine.trace(poutine.replay(model, trace=guide_trace), graph_type=graph_type).get_trace(*args, **kwargs)

    if is_validation_enabled():
        check_model_guide_match(model_trace, guide_trace, max_plate_nesting)

    guide_trace = prune_subsample_sites(guide_trace)
    model_traces = prune_subsample_sites(model_trace)

    return model_traces, guide_trace

class MSE(ELBO):
    def __init__(self,
                 observed_sitename,
                 observed_dataname,
                 num_particles=2,
                 max_plate_nesting=float('inf'),
                 max_iarange_nesting=None,  # DEPRECATED
                 vectorize_particles=False,
                 strict_enumeration_warning=True):
        if max_iarange_nesting is not None:
            warnings.warn("max_iarange_nesting is deprecated; use max_plate_nesting instead",
                          DeprecationWarning)
            max_plate_nesting = max_iarange_nesting

        self.observed_sitename = observed_sitename
        self.observed_dataname = observed_dataname

        super(MSE, self).__init__(num_particles=num_particles,
                                        max_plate_nesting=max_plate_nesting,
                                        vectorize_particles=vectorize_particles,
                                        strict_enumeration_warning=strict_enumeration_warning)

    def _get_trace(self, model, guide, *args, **kwargs):
        """
        Returns a single trace from the guide, and the model that is run
        against it.
        """
        model_trace, guide_trace = get_importance_trace("flat", self.max_plate_nesting, model, guide, *args, **kwargs)
        return model_trace, guide_trace

    def _get_traces(self, model, guide, *args, **kwargs):
        """
        Runs the guide and runs the model against the guide with
        the result packaged as a trace generator.
        """
        if self.vectorize_particles:
            if self.max_plate_nesting == float('inf'):
                self._guess_max_plate_nesting(model, guide, *args, **kwargs)
            yield self._get_vectorized_trace(model, guide, *args, **kwargs)
        else:
            for i in range(self.num_particles):
                yield self._get_trace(model, guide, *args, **kwargs)

    def loss(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Evaluates the ELBO with an estimator that uses num_particles many samples/particles.
        """

        is_vectorized = self.vectorize_particles and self.num_particles > 1

        # grab a vectorized trace from the generator
        data = args[0].copy()
        observed_truth = data[self.observed_dataname]
        data[self.observed_dataname] = None
        args = [data] + list(args[1:])
        for model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):

            observed_samples = model_trace.nodes[self.observed_sitename]["value"]
            loss = (observed_samples - observed_truth).pow(2).mean()

        warn_if_nan(loss, "loss")
        return loss.item()
