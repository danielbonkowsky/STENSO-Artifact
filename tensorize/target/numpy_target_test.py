from utils import *


prog_str = """
def fn_0(arg0):
    x0 = arg0
    return x0
"""
print(inline_stmts(prog_str))


prog_str2 = """
def fn_0(arg0, arg1):
    x0 = jnp.transpose(arg0)
    x1 = jnp.multiply(x0, arg1)
    x2 = jnp.sum(x1, axis=0)
    return x2
"""
print(inline_stmts(prog_str2))

prog_str3 = """
def fn_0(arg0):
    x0 = jnp.full([3, 3], 5.0)
    x1 = jnp.power(arg0, x0)
    return x1

"""
print(replace_jnp_full(inline_stmts(prog_str3)))