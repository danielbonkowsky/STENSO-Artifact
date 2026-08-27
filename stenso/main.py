import sys
import json
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from absl import app

import mlir_synth.synth as mlir_synth
import target.numpy_target as numpy_target
from target.utils import *
from mlir_synth.ir import *
from mlir_synth.passmanager import *
from mlir_helpers import *
from synthesis import *

from utils import *

FLAGS = flags.FLAGS
flags.DEFINE_string("program", None, "Path to the program to synthesize")
flags.DEFINE_string("synth_out", "", "Result file containing synthesized program")

flags.DEFINE_bool("no_branch_and_bound", False, "Don't do branch and bound")
flags.DEFINE_string(
    "cost_estimator", "flops", "Cost estimator to use: flops, measured, measuring"
)

flags.DEFINE_boolean("debug", False, "Debug mode")
flags.DEFINE_list("ops", [], "List of operations to use")


def estimate_cost(source_function, args):
    jax_fn = jax.jit(source_function)
    lowered = jax_fn.lower(*args)

    # Get HLO
    hlo = lowered.compiler_ir(dialect="mhlo")
    if len(list(hlo.body.operations)) > 1:
        hlo = inline_functions(hlo)
    hlo_str = str(hlo)

    # Get cost
    compiled = lowered.compile()
    analysis = compiled.cost_analysis()

    flops = 0
    if "flops" in analysis[0]:
        # assert len(analysis) == 1
        flops = analysis[0]["flops"]
    return flops


def main(argv):
    # Load the input program into the global namespace
    with open(FLAGS.program, "r") as f:
        program_contents = f.read()
    exec(program_contents, globals())

    if "kernel" not in globals():
        raise ValueError("No function named 'kernel' found in the program.")
    if "arg_ranks" not in globals():
        raise ValueError("No variable named 'arg_ranks' found in the program.")

    target = numpy_target.NumpyTarget()

    args = arg_ranks_to_args(arg_ranks)
    nops = count_nops(get_kernel_function(program_contents))

    original_cost = estimate_cost(kernel, args)
    print_green("Original cost: %f" % original_cost)

    cost_estimator = None
    if FLAGS.cost_estimator == "flops":
        cost_estimator = FlopsCostEstimator()
    elif FLAGS.cost_estimator == "measuring":
        cost_estimator = MeasuringCostEstimator()
    else:
        raise ValueError("Unknown cost estimator: %s" % FLAGS.cost_estimator)

    with Context():
        mlir_synth.register_dialects()
        mlir_synth.register_passes()

        jax_kernel = jax.jit(kernel, backend="cpu", keep_unused=True)
        mlir_high_level_str = str(jax_kernel.lower(*args).compiler_ir(dialect="mhlo"))

        mod_hl = Module.parse(mlir_high_level_str)

        mlir_synth.lower_chlo_to_affine(mod_hl, False)

        for func_idx, func in enumerate(mod_hl.body.operations):
            ops = []
            if FLAGS.ops:
                ops = FLAGS.ops

            raised_fn, cost = synthesize(
                func,
                ops,
                target,
                original_cost,
                cost_estimator,
                nops,
            )

            print_green("Cost: %f" % cost)

            prog_opt = target.construct_function_ast(
                raised_fn, func, func_idx, "kernel"
            )
            prog_opt = inline_stmts(prog_opt)
            prog_opt = replace_jnp_full(prog_opt)
            print_green(prog_opt)

            if FLAGS.synth_out:
                with open(FLAGS.synth_out, "w") as f:
                    f.write(prog_opt)

    print_stats()

    print()
    print("JSON:", json.dumps(all_stats))


if __name__ == "__main__":
    app.run(main)
