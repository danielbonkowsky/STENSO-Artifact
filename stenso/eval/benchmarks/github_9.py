import jax.numpy as jnp

arg_ranks = [1]


def kernel(A):
    return jnp.sum(jnp.power(A, 2), axis=-1)
