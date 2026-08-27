import jax.numpy as jnp

arg_ranks = [2]


def kernel(A):
    return jnp.sum(jnp.sum(A, axis=0), axis=0)
