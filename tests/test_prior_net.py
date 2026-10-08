"""Minimal prior net: shape, positive precision, finite synthetic updates."""

import jax
import jax.numpy as jnp
import optax

from shape_bayes.loss import shape_bayes_loss
from shape_bayes.posterior import observation_precision_diag, posterior_mean
from shape_bayes.prior_net import init_prior_net, prior_logits, prior_precision_diag


def test_logits_shape_and_positive_precision():
    k = 3
    n_landmarks = 5
    params = init_prior_net(jax.random.PRNGKey(0), k)
    tokens = jax.random.normal(jax.random.PRNGKey(1), (n_landmarks, 4))
    logits = prior_logits(params, tokens)
    lam = prior_precision_diag(params, tokens)
    assert logits.shape == (k,)
    assert lam.shape == (k,)
    assert jnp.all(jnp.exp(logits) > 0)
    assert jnp.allclose(lam, jnp.exp(logits))
    assert jnp.all(lam > 0)


def test_five_adam_steps_stay_finite():
    k = 3
    n_landmarks = 4
    batch = 4
    two_n = 2 * n_landmarks
    keys = jax.random.split(jax.random.PRNGKey(7), 6)
    params = init_prior_net(keys[0], k)
    tokens = jax.random.normal(keys[1], (batch, n_landmarks, 4))
    b_gt = jax.random.normal(keys[2], (batch, k)) * 0.1
    basis = jax.random.normal(keys[3], (two_n, k))
    P, _ = jnp.linalg.qr(basis)
    Sbar = jax.random.normal(keys[4], (two_n,)) * 0.1
    s_prime = Sbar + P @ jax.random.normal(keys[5], (k,)) * 0.1
    sigma = jnp.full((two_n,), 0.5)
    eps = 1e-6
    w_diag = observation_precision_diag(sigma, eps=eps)

    def loss_fn(theta):
        def one(tok, coeff):
            lam = prior_precision_diag(theta, tok)
            post = posterior_mean(P, Sbar, s_prime, w_diag, lam)
            return shape_bayes_loss(post.b_mu, coeff, post.prec, lam_mse=1.0, lam_nll=0.1)

        return jnp.mean(jax.vmap(one)(tokens, b_gt))

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    for _step in range(5):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        assert jnp.isfinite(loss)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
    final = loss_fn(params)
    assert jnp.isfinite(final)
