"""Minimal stand-in for the conditional prior network E_phi.

Not the unpublished architecture. One transformer block:

    tokens (N, 4) = [x, y, sigma_x, sigma_y]
    linear embed to d_model = 32
    2-head self-attention
    residual add onto the embedding
    mean-pool over landmarks
    linear map to K logits

Lambda_prior = diag(exp(logits)). Parameters are a pytree of arrays, initialized
from a PRNG key. No Flax and no Equinox.
"""

import jax
import jax.numpy as jnp

from shape_bayes.posterior import prior_precision_from_logits

D_MODEL = 32
N_HEADS = 2
IN_DIM = 4


def _glorot(key, din, dout):
    std = jnp.sqrt(2.0 / (din + dout))
    return jax.random.normal(key, (din, dout)) * std


def init_prior_net(key, num_components, d_model=D_MODEL, in_dim=IN_DIM):
    """Initialize E_phi parameters. num_components is K, the PCA dimension."""
    if d_model % N_HEADS != 0:
        raise ValueError("d_model must be divisible by 2")
    k_embed_w, k_wq, k_wk, k_wv, k_wo, k_head_w = jax.random.split(key, 6)
    return {
        "embed": {
            "w": _glorot(k_embed_w, in_dim, d_model),
            "b": jnp.zeros((d_model,)),
        },
        "attn": {
            "wq": _glorot(k_wq, d_model, d_model),
            "wk": _glorot(k_wk, d_model, d_model),
            "wv": _glorot(k_wv, d_model, d_model),
            "wo": _glorot(k_wo, d_model, d_model),
        },
        "head": {
            "w": _glorot(k_head_w, d_model, num_components),
            "b": jnp.zeros((num_components,)),
        },
    }


def _split_heads(projected, n_heads):
    n_tokens, d_model = projected.shape
    head_dim = d_model // n_heads
    heads = projected.reshape(n_tokens, n_heads, head_dim)
    return jnp.swapaxes(heads, 0, 1)


def _self_attention(attn, hidden):
    """2-head self-attention. hidden is (N, d_model)."""
    q = _split_heads(hidden @ attn["wq"], N_HEADS)
    k = _split_heads(hidden @ attn["wk"], N_HEADS)
    v = _split_heads(hidden @ attn["wv"], N_HEADS)
    head_dim = q.shape[-1]
    scale = jnp.sqrt(jnp.asarray(head_dim, dtype=hidden.dtype))
    scores = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) / scale
    weights = jax.nn.softmax(scores, axis=-1)
    mixed = jnp.matmul(weights, v)
    mixed = jnp.swapaxes(mixed, 0, 1)
    n_tokens = hidden.shape[0]
    merged = mixed.reshape(n_tokens, hidden.shape[-1])
    return merged @ attn["wo"]


def prior_logits(params, tokens):
    """Map landmark tokens (N, 4) to prior logits of shape (K,)."""
    tokens = jnp.asarray(tokens)
    hidden = tokens @ params["embed"]["w"] + params["embed"]["b"]
    hidden = hidden + _self_attention(params["attn"], hidden)
    pooled = jnp.mean(hidden, axis=0)
    return pooled @ params["head"]["w"] + params["head"]["b"]


def prior_precision_diag(params, tokens):
    """Diagonal of Lambda_prior = diag(exp(logits))."""
    return prior_precision_from_logits(prior_logits(params, tokens))
