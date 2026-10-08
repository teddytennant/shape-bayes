"""Shape-Bayes training loss.

    L = lam_mse * ||b_mu - b_gt||**2 + lam_nll * L_nll
    L_nll = 0.5 * (b_gt - stop_gradient(b_mu))**T @ prec @ (b_gt - stop_gradient(b_mu))
            - 0.5 * logdet(prec)

stop_gradient is applied to b_mu inside the NLL only. The squared-error term
still depends on b_mu. logdet is the slogdet log-absolute-determinant, which
equals log(det) when prec is symmetric positive definite. The 0.5 * K * log(2
pi) constant is not part of the given NLL.
"""

import jax
import jax.numpy as jnp


def nll_loss(b_gt, b_mu, prec):
    """Gaussian NLL of b_gt under precision prec, with b_mu stopped."""
    b_gt = jnp.asarray(b_gt)
    b_mu = jnp.asarray(b_mu)
    prec = jnp.asarray(prec)
    diff = b_gt - jax.lax.stop_gradient(b_mu)
    quad = jnp.dot(diff, prec @ diff)
    _sign, logdet = jnp.linalg.slogdet(prec)
    return 0.5 * quad - 0.5 * logdet


def mse_coeff(b_mu, b_gt):
    """Squared Euclidean error ||b_mu - b_gt||**2. Gradients flow through b_mu."""
    err = jnp.asarray(b_mu) - jnp.asarray(b_gt)
    return jnp.dot(err, err)


def shape_bayes_loss(b_mu, b_gt, prec, lam_mse=1.0, lam_nll=1.0):
    """Weighted sum of coefficient MSE and stopped-mean NLL."""
    return lam_mse * mse_coeff(b_mu, b_gt) + lam_nll * nll_loss(b_gt, b_mu, prec)
