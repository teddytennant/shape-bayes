"""NLL value, stop-gradient, and live MSE gradient."""

import jax
import jax.numpy as jnp

from shape_bayes.loss import mse_coeff, nll_loss, shape_bayes_loss


def _fixture():
    b_gt = jnp.array([0.3, -0.4])
    b_mu = jnp.array([0.1, 0.2])
    prec = jnp.array([[2.0, 0.3], [0.3, 1.5]])
    return b_gt, b_mu, prec


def test_nll_matches_formula():
    b_gt, b_mu, prec = _fixture()
    diff = b_gt - b_mu
    quad = diff @ prec @ diff
    _sign, logdet = jnp.linalg.slogdet(prec)
    expected = 0.5 * quad - 0.5 * logdet
    assert jnp.allclose(nll_loss(b_gt, b_mu, prec), expected)
    assert jnp.isfinite(nll_loss(b_gt, b_mu, prec))


def test_stop_gradient_kills_nll_grad_but_not_mse():
    b_gt, b_mu, prec = _fixture()
    shift0 = jnp.zeros_like(b_mu)

    def nll_of_shift(shift):
        return nll_loss(b_gt, b_mu + shift, prec)

    def mse_of_shift(shift):
        return mse_coeff(b_mu + shift, b_gt)

    def total_of_shift(shift):
        return shape_bayes_loss(b_mu + shift, b_gt, prec, lam_mse=1.0, lam_nll=1.0)

    grad_nll = jax.grad(nll_of_shift)(shift0)
    grad_mse = jax.grad(mse_of_shift)(shift0)
    grad_total = jax.grad(total_of_shift)(shift0)

    assert jnp.allclose(grad_nll, jnp.zeros_like(shift0))
    assert not jnp.allclose(grad_mse, jnp.zeros_like(shift0))
    assert not jnp.allclose(grad_total, jnp.zeros_like(shift0))
    # The combined loss still sees the MSE path, so its shift gradient matches MSE.
    assert jnp.allclose(grad_total, grad_mse)
