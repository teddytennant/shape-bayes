"""RMS uncertainty target is not the ensemble standard deviation."""

import jax.numpy as jnp

from shape_bayes.uncertainty import sigma_target


def test_sigma_target_is_rms_against_unaugmented_target():
    teachers = jnp.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 0.0, 0.0, 5.0],
        ]
    )
    s_target = jnp.array([0.5, 0.5, 0.5, 0.5])
    manual = jnp.sqrt(jnp.mean((teachers - s_target) ** 2, axis=0))
    got = sigma_target(teachers, s_target)
    assert got.shape == s_target.shape
    assert jnp.allclose(got, manual)

    ensemble_mean = jnp.mean(teachers, axis=0)
    ensemble_std = jnp.sqrt(jnp.mean((teachers - ensemble_mean) ** 2, axis=0))
    assert not jnp.allclose(s_target, ensemble_mean)
    assert not jnp.allclose(got, ensemble_std)


def test_sigma_target_matches_ensemble_std_only_when_target_is_mean():
    teachers = jnp.array(
        [
            [0.0, 2.0],
            [2.0, 0.0],
            [4.0, 4.0],
        ]
    )
    ensemble_mean = jnp.mean(teachers, axis=0)
    ensemble_std = jnp.std(teachers, axis=0)
    assert jnp.allclose(sigma_target(teachers, ensemble_mean), ensemble_std)
