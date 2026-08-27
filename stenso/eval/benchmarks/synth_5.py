import jax.numpy as jnp

arg_ranks = [2, 2]


def kernel(A, B):
    return jnp.power(jnp.sqrt(A), 4) + 2 * B
