import jax.numpy as jnp

arg_ranks = [2, 2]


def kernel(A, B):
    return jnp.sum(jnp.diag(jnp.dot(A, B)))
