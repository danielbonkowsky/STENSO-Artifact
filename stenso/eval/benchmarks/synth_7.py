import jax.numpy as jnp

arg_ranks = [2]


def kernel(A):
    return jnp.power(A, 6) / jnp.power(A, 4)
