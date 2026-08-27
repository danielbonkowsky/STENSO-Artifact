import jax.numpy as jnp

arg_ranks = [2, 1, 1]


def kernel(A, x, y):
    return jnp.stack([(x * a + (1.0 - a) * y) for a in A], axis=0)
