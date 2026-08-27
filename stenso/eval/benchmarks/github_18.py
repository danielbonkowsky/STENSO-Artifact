import jax.numpy as jnp

arg_ranks = [2, 2, 2]


def kernel(A, B, C):
    return jnp.sum(jnp.stack([A, B, C], axis=0), axis=0)
