"""Uncertainty target from teacher shapes.

Given teacher shapes Si, i = 1..M, each of shape (2N,), and an unaugmented
target S_target of shape (2N,),

    sigma_target = sqrt(mean_i (Si - S_target)**2)

elementwise over coordinates. This is the RMS deviation from the unaugmented
target, not the standard deviation around the ensemble mean. If S_target is
not the mean of the Si, the two quantities differ.
"""

import jax.numpy as jnp


def sigma_target(teacher_shapes, s_target):
    """RMS deviation of teacher shapes from the unaugmented target.

    teacher_shapes: (M, 2N)
    s_target: (2N,)
    returns: (2N,)
    """
    teachers = jnp.asarray(teacher_shapes)
    target = jnp.asarray(s_target)
    deviation = teachers - target
    return jnp.sqrt(jnp.mean(jnp.square(deviation), axis=0))
