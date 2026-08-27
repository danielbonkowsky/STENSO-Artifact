import jax.numpy as jnp

arg_ranks = [3, 2]


def kernel(A, B):
    NR, NQ, NP = A.shape[0], A.shape[1], B.shape[1]
    return jnp.reshape(jnp.reshape(A, (NR, NQ, 1, NP)) @ B, (NR, NQ, NP))
