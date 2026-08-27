import jax.numpy as jnp

arg_ranks = [0, 2, 2]


def kernel(A, B, C):
    return A * B @ C
