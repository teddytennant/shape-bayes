"""20 Adam steps of the shape posterior loss on a synthetic batch. 1x H200 smoke."""

import jax
import jax.numpy as jnp
import optax

from shape_bayes.loss import shape_bayes_loss
from shape_bayes.posterior import observation_precision_diag, posterior_mean
from shape_bayes.prior_net import init_prior_net, prior_precision_diag


def main():
    print("jax", jax.__version__, jax.devices())
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
    w_diag = observation_precision_diag(sigma, eps=1e-6)

    def loss_fn(theta):
        def one(tok, coeff):
            lam = prior_precision_diag(theta, tok)
            post = posterior_mean(P, Sbar, s_prime, w_diag, lam)
            return shape_bayes_loss(post.b_mu, coeff, post.prec, lam_mse=1.0, lam_nll=0.1)

        return jnp.mean(jax.vmap(one)(tokens, b_gt))

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    initial = None
    loss = None
    for step in range(20):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        if not bool(jnp.isfinite(loss)):
            raise SystemExit(f"non-finite loss at step {step}: {loss}")
        if initial is None:
            initial = float(loss)
        print(f"step {step} loss {float(loss):.6f}")
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
    print(f"initial {initial:.6f} final {float(loss):.6f}")
    print("shape-bayes gpu smoke done")


if __name__ == "__main__":
    main()
