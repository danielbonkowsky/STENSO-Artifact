import jax.numpy as jnp

arg_ranks = [2, 2]


def kernel(A, B):
    return (A * B) + 3 * (A * B)
