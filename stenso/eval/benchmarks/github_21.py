import jax.numpy as jnp

arg_ranks = [2, 2]


def kernel(A, B):
    return jnp.max(jnp.stack([A, B], axis=0), axis=0)
