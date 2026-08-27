import jax.numpy as jnp

arg_ranks = [2, 2]


def kernel(A, B):
    return jnp.trace(A @ B.T)
