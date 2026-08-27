import jax.numpy as jnp

arg_ranks = [2]


def kernel(A):
    return jnp.power(A, 2)
