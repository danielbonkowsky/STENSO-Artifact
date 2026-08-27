import jax.numpy as jnp

arg_ranks = [2, 2]


def kernel(A, B):
    return A + A + B - A - A - B * B
