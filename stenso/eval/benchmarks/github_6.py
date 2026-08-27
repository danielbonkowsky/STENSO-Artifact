import jax.numpy as jnp

arg_ranks = [2, 1]


def kernel(A, x):
    return jnp.dot(A.T, x.T)
