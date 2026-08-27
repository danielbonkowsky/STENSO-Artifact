import jax.numpy as jnp

arg_ranks = [1, 1]


def kernel(a, b):
    return jnp.sum(a * b)
