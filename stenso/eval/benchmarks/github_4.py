import jax.numpy as jnp

arg_ranks = [2, 2]


def kernel(A, B):
    return jnp.exp(jnp.log(A) - jnp.log(B))
