from absl import app, flags
import jax
import json
import time

jax.config.update("jax_enable_x64", True)

import mlir_synth.synth as mlir_synth
import target.numpy_target as numpy_target
from mlir_helpers import *
from synthesis import *

from utils import *

FLAGS = flags.FLAGS
flags.DEFINE_string("program", None, "Path to the program to synthesize")


def main(argv):
    # Load the input program into the global namespace
    with open(FLAGS.program, "r") as f:
        program_contents = f.read()
    exec(program_contents, globals())

    if "kernel" not in globals():
        raise ValueError("No function named 'kernel' found in the program.")
    if "arg_ranks" not in globals():
        raise ValueError("No variable named 'arg_ranks' found in the program.")

    args = arg_ranks_to_args(arg_ranks)
    nops = count_nops(get_kernel_function(program_contents))

    start_time = time.time()
    with Context():
        mlir_synth.register_dialects()
        mlir_synth.register_passes()

        jax_kernel = jax.jit(kernel, backend="cpu", keep_unused=True)
        mlir_high_level_str = str(jax_kernel.lower(*args).compiler_ir(dialect="mhlo"))

        mod_hl = Module.parse(mlir_high_level_str)
        mlir_synth.lower_chlo_to_affine(mod_hl, False)

        for func_idx, func in enumerate(mod_hl.body.operations):
            source_module = "module { %s }" % str(func)
            arg_symbols = affine_to_arg_symbols(source_module)
            source_program = affine_to_python(source_module)
            source_expression = run(source_program, copy_arg_symbols(arg_symbols))
            constants = get_constants(func, source_expression)

            target = numpy_target.NumpyTarget()
            source_module = "module { %s }" % str(func)
            stubs = target.get_stubs(source_module, nops, [], constants, True)

    end_time = time.time()

    print()
    print(
        "JSON:",
        json.dumps(
            {
                "synthesis_time": end_time - start_time,
            }
        ),
    )


if __name__ == "__main__":
    app.run(main)
