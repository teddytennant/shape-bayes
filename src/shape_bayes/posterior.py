"""Closed-form PCA shape posterior.

Landmark coordinates of shape (N, 2) are flattened to length 2N in row-major
order: x0, y0, x1, y1, ..., x_{N-1}, y_{N-1}. The shape manifold is

    S = Sbar + P @ b

with Sbar of shape (2N,), P of shape (2N, K), and b of shape (K,).

The two-column PDF places a superscript 2 on the line above the epsilon, so
the observation precision is

    W = diag(1 / (sigma**2 + eps))

and not 1 / (sigma + eps). sigma is the base model's per-coordinate aleatoric
uncertainty and has the same shape as S.

The conditional prior on b is zero-mean with precision Lambda_prior =
diag(exp(logits)). The posterior precision and mean are

    prec = P.T @ diag(W) @ P + Lambda_prior
    rhs = P.T @ (W_diag * (s_prime - Sbar))
    b_mu = solve(prec, rhs)
    S_post = Sbar + P @ b_mu

prec is not inverted explicitly.
"""

from typing import NamedTuple

import jax.numpy as jnp


class Posterior(NamedTuple):
    """Posterior coefficient mean, reconstructed shape, and precision of b."""

    b_mu: jnp.ndarray
    s_post: jnp.ndarray
    prec: jnp.ndarray


def flatten_landmarks(landmarks):
    """Flatten (..., N, 2) landmarks to (..., 2N,) as x0, y0, x1, y1, ..."""
    coords = jnp.asarray(landmarks)
    if coords.shape[-1] != 2:
        raise ValueError("last axis must be 2 (x, y)")
    n = coords.shape[-2]
    return coords.reshape(coords.shape[:-2] + (2 * n,))


def observation_precision_diag(sigma, eps=1e-6):
    """Per-coordinate observation precision 1 / (sigma**2 + eps).

    The superscript 2 sits on sigma, not on the whole denominator. eps is
    added after the square.
    """
    sigma = jnp.asarray(sigma)
    return 1.0 / (jnp.square(sigma) + eps)


def prior_precision_from_logits(logits):
    """Diagonal of Lambda_prior = diag(exp(logits)). Positive by construction."""
    return jnp.exp(jnp.asarray(logits))


def posterior_precision(P, w_diag, lambda_diag):
    """Posterior precision of b, shape (K, K).

    prec = P.T @ diag(w_diag) @ P + diag(lambda_diag).
    A zero observation weight leaves prec equal to diag(lambda_diag).
    """
    P = jnp.asarray(P)
    w_diag = jnp.asarray(w_diag)
    lambda_diag = jnp.asarray(lambda_diag)
    # (P.T * w_diag) @ P equals P.T @ diag(w_diag) @ P without building W.
    gram = (P.T * w_diag) @ P
    return gram + jnp.diag(lambda_diag)


def posterior_mean(P, Sbar, s_prime, w_diag, lambda_diag):
    """Closed-form posterior mean of b and the reconstructed shape.

    The prior mean of b is 0. Pass w_diag directly (use a zero vector for a
    flat observation). lambda_diag is the diagonal of Lambda_prior, not the
    full matrix. b_mu is obtained with a linear solve, not an explicit inverse.
    """
    P = jnp.asarray(P)
    Sbar = jnp.asarray(Sbar)
    s_prime = jnp.asarray(s_prime)
    w_diag = jnp.asarray(w_diag)
    lambda_diag = jnp.asarray(lambda_diag)
    prec = posterior_precision(P, w_diag, lambda_diag)
    rhs = P.T @ (w_diag * (s_prime - Sbar))
    b_mu = jnp.linalg.solve(prec, rhs)
    s_post = Sbar + P @ b_mu
    return Posterior(b_mu=b_mu, s_post=s_post, prec=prec)
