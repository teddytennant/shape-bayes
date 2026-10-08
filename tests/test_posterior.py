"""Hand checks for the closed-form shape posterior."""

import jax.numpy as jnp

from shape_bayes.posterior import (
    flatten_landmarks,
    observation_precision_diag,
    posterior_mean,
)


def _small_system():
    P = jnp.array(
        [
            [1.0, 0.2],
            [0.1, 1.0],
            [-0.3, 0.4],
            [0.5, -0.2],
        ]
    )
    Sbar = jnp.array([0.1, -0.2, 0.3, 0.0])
    s_prime = jnp.array([1.0, 0.5, -0.2, 0.8])
    w_diag = jnp.array([2.0, 0.5, 1.5, 3.0])
    lambda_diag = jnp.array([0.4, 0.7])
    return P, Sbar, s_prime, w_diag, lambda_diag


def test_observation_precision_uses_sigma_squared():
    sigma = jnp.array([2.0, 0.5, 3.0, 1.0])
    eps = 1e-6
    weights = observation_precision_diag(sigma, eps=eps)
    expected = 1.0 / (sigma**2 + eps)
    not_this = 1.0 / (sigma + eps)
    assert jnp.allclose(weights, expected)
    assert not jnp.allclose(weights, not_this)


def test_flatten_landmarks_is_row_major():
    landmarks = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    flat = flatten_landmarks(landmarks)
    assert jnp.allclose(flat, jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


def test_zero_observation_weight_returns_prior():
    P, Sbar, s_prime, _w_diag, lambda_diag = _small_system()
    w_diag = jnp.zeros_like(s_prime)
    post = posterior_mean(P, Sbar, s_prime, w_diag, lambda_diag)
    assert post.b_mu.shape == lambda_diag.shape
    assert jnp.allclose(post.b_mu, jnp.zeros_like(lambda_diag))
    assert jnp.allclose(post.prec, jnp.diag(lambda_diag))
    assert jnp.allclose(post.s_post, Sbar)


def test_posterior_matches_hand_built_solve():
    P, Sbar, s_prime, w_diag, lambda_diag = _small_system()
    prec = P.T @ jnp.diag(w_diag) @ P + jnp.diag(lambda_diag)
    rhs = P.T @ (w_diag * (s_prime - Sbar))
    b_hat = jnp.linalg.solve(prec, rhs)
    post = posterior_mean(P, Sbar, s_prime, w_diag, lambda_diag)
    assert post.b_mu.shape == (2,)
    assert post.s_post.shape == (4,)
    assert jnp.allclose(post.b_mu, b_hat, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(post.prec, prec, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(post.s_post, Sbar + P @ post.b_mu)
    assert jnp.allclose(post.s_post, Sbar + P @ b_hat, rtol=1e-5, atol=1e-5)


def test_large_weight_near_least_squares():
    P, Sbar, s_prime, _w_diag, _lambda_diag = _small_system()
    w_diag = jnp.full_like(s_prime, 1e4)
    lambda_diag = jnp.array([1e-6, 1e-6])
    gram = P.T @ jnp.diag(w_diag) @ P
    rhs = P.T @ (w_diag * (s_prime - Sbar))
    b_ls = jnp.linalg.solve(gram, rhs)
    post = posterior_mean(P, Sbar, s_prime, w_diag, lambda_diag)
    assert jnp.allclose(post.b_mu, b_ls, rtol=1e-4, atol=1e-4)
