import jax.numpy as jnp

arg_ranks = [2]


def kernel(A):
    return jnp.stack([x * 2 for x in A], axis=0)
