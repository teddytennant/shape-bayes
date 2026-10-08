"""Closed-form Bayesian 2D shape posterior (arXiv 2610.09032)."""

from shape_bayes.loss import mse_coeff, nll_loss, shape_bayes_loss
from shape_bayes.posterior import (
    Posterior,
    flatten_landmarks,
    observation_precision_diag,
    posterior_mean,
    posterior_precision,
    prior_precision_from_logits,
)
from shape_bayes.prior_net import init_prior_net, prior_logits, prior_precision_diag
from shape_bayes.uncertainty import sigma_target

__all__ = [
    "Posterior",
    "flatten_landmarks",
    "init_prior_net",
    "mse_coeff",
    "nll_loss",
    "observation_precision_diag",
    "posterior_mean",
    "posterior_precision",
    "prior_logits",
    "prior_precision_diag",
    "prior_precision_from_logits",
    "shape_bayes_loss",
    "sigma_target",
]
