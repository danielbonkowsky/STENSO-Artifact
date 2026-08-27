import numpy as np
import jax
import jax.numpy as jnp
import torch
import cpuinfo
from absl import app
from absl import flags

import time
import os
import subprocess
import re

from utils import *

FLAGS = flags.FLAGS
flags.DEFINE_string("out_dir", "out", "Output dir")
flags.DEFINE_string(
    "benchmark_dir", "out/benchmarks_merged", "Directory containing benchmark files"
)
flags.DEFINE_bool("skip_csv_logging", False, "Skip logging to CSV")

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def get_processor_name():
    """
    Retrieves the processor's vendor and model name in a formatted string.
    Returns:
        str: A formatted string like "Intel_Core_i7-8700K" or
             "AMD_Ryzen_9_7950X", or "Unknown_Processor" if
             the information cannot be retrieved.
    """
    try:
        info = cpuinfo.get_cpu_info()
        vendor = info.get("vendor_id_raw", "UnknownVendor")
        model = info.get("brand_raw", "UnknownModel")

        if "amd" in vendor.lower():
            vendor = "AMD"
        elif "intel" in vendor.lower():
            vendor = "Intel"

        model = model.replace("(R)", "").replace("(TM)", "")
        junk_terms = [
            "CPU",
            "Processor",
            "APU with Radeon Graphics",
            "8-Core",
            "16-Core",
            "6-Core",
            "4-Core",
            "Dual-Core",
            "AMD",
            "Intel",
        ]
        for term in junk_terms:
            model = re.sub(r"\b" + term + r"\b", "", model, flags=re.IGNORECASE)

        model = model.split("@")[0]
        model = " ".join(model.split())
        model_formatted = model.replace(" ", "_")
        model_formatted = re.sub(r"_\d+$", "", model_formatted)

        if not model_formatted:
            model_formatted = "UnknownModel"

        return f"{vendor}_{model_formatted}"

    except Exception as e:
        print(f"An error occurred: {e}")
        return "Unknown_Processor"


CSV_FILE = os.path.join(
    SCRIPT_DIR, "performance_results", get_processor_name() + ".csv"
)
os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)


def resize_arg_shapes(args) -> jnp.ndarray:
    """
    Create a new array with random values and for each dimension, the size is N.
    """
    array_args = [arg for arg in args if hasattr(arg, "shape")]
    all_rank_smaller_two = all(arg.ndim < 2 for arg in array_args)
    N = 1000 if all_rank_smaller_two else 200

    new_args = []
    for arg in args:
        if hasattr(arg, "shape"):  # It's an array
            new_shape = tuple(N if dim > 0 else 1 for dim in arg.shape)
            # Preserve the original dtype
            if arg.dtype == bool or arg.dtype == torch.bool:
                new_arg = np.random.choice([True, False], size=new_shape)
            else:
                if new_shape == ():
                    new_arg = np.random.rand(*new_shape)
                else:
                    new_arg = np.random.rand(*new_shape).astype(np.float32)
            new_args.append(new_arg)
        else:  # It's a scalar (int, float, etc.)
            new_args.append(arg)
    return new_args


def bench(fn, args, num_runs=100):
    elapsed_times = []
    for _ in range(num_runs):
        start = time.time()
        fn(*args)
        elapsed_time = time.time() - start
        elapsed_times.append(elapsed_time)
    return np.median(elapsed_times)


def validate(fn_1, fn_2, args):
    """
    Validate that two functions produce the same output for the given arguments.
    """
    result_1 = fn_1(*args)
    result_2 = fn_2(*args)
    if not jnp.allclose(result_1, result_2):
        print(result_1)
        print(result_2)
        raise ValueError("Outputs of the two functions do not match!")


def write_csv_header():
    if not FLAGS.skip_csv_logging:
        with open(CSV_FILE, "w") as f:
            f.write("Benchmark,Compiler,OriginalTime,OptimizedTime\n")


def write_csv_row(benchmark, variant, compiler, time):
    if not FLAGS.skip_csv_logging:
        with open(CSV_FILE, "a") as f:
            f.write(f"{benchmark},{variant},{compiler},{time}\n")


def fetch_kernel_and_arg_ranks_from_globals():
    kernel = globals().get("kernel")
    kernel_opt = globals().get("kernel_stenso")
    arg_ranks = globals().get("arg_ranks", [])
    return kernel, kernel_opt, arg_ranks


def compatibalize_dot_at(program_contents):
    if ".at[" not in program_contents:
        return program_contents

    # Remove "variable = " from the beginning of .at[] lines
    program_contents = re.sub(
        r"(\s*)(\w+)\s*=\s*(\w+)\.at\[", r"\1\2[", program_contents
    )

    # Replace ].set( with ] =
    program_contents = program_contents.replace("].set(", "] = ")

    # Remove the extra closing parenthesis at the end of these lines
    lines = program_contents.split("\n")
    for i, line in enumerate(lines):
        if "] = " in line and line.strip().endswith(")"):
            # Remove last closing parenthesis
            lines[i] = line.rsplit(")", 1)[0]

    program_contents = "\n".join(lines)
    return program_contents


def compatibalize_transpose(program_contents):
    # Replace torch.transpose(arg) with arg.T
    program_contents = re.sub(r"torch.transpose\((\w+)\)", r"\1.T", program_contents)
    return program_contents


def main(argv):
    # Cleanup previous CSV file
    if os.path.exists(CSV_FILE):
        os.remove(CSV_FILE)

    # Run for all files in the benchmarks directory
    benchmark_files = os.listdir(FLAGS.benchmark_dir)

    benchmark_files.sort(
        key=lambda x: (
            x.split(".")[0].split("_")[0],
            int(x.split(".")[0].split("_")[-1]),
        )
    )

    for benchmark_file in benchmark_files:
        if not os.path.exists(CSV_FILE):
            write_csv_header()
        print(benchmark_file)
        benchmark = benchmark_file.split(".")[0]

        with open(os.path.join(FLAGS.benchmark_dir, benchmark_file), "r") as f:
            program_contents = f.read()

        # JAX
        # ----------------
        exec(program_contents, globals())
        kernel, kernel_opt, arg_ranks = fetch_kernel_and_arg_ranks_from_globals()
        args = arg_ranks_to_args(arg_ranks)

        # Validate
        validate(kernel, kernel_opt, args)

        # Benchmark
        args = resize_arg_shapes(args)
        args = [jnp.asarray(arg) for arg in args]

        kernel = jax.jit(kernel)
        kernel_opt = jax.jit(kernel_opt)

        time_orig = bench(kernel, args)
        time_opt = bench(kernel_opt, args)

        speedup = time_orig / time_opt

        print(f"Benchmark (Jax): {benchmark_file}")
        print(f"Original time: {time_orig:.6f} seconds")
        print(f"Optimized time: {time_opt:.6f} seconds")
        print(f"Speedup: {speedup:.2f}x")
        print()

        write_csv_row(benchmark, "JAX", time_orig, time_opt)

        # NumPy
        # ----------------
        program_contents_np = program_contents.replace("jnp.", "np.")
        program_contents_np = compatibalize_dot_at(program_contents_np)
        exec(program_contents_np, globals())
        kernel, kernel_opt, args = fetch_kernel_and_arg_ranks_from_globals()
        args = arg_ranks_to_args(arg_ranks)

        # Validate
        validate(kernel, kernel_opt, args)

        # Benchmark
        args = resize_arg_shapes(args)
        time_orig = bench(kernel, args)
        time_opt = bench(kernel_opt, args)

        speedup = time_orig / time_opt

        print(f"Benchmark (Numpy): {benchmark_file}")
        print(f"Original time: {time_orig:.6f} seconds")
        print(f"Optimized time: {time_opt:.6f} seconds")

        print(f"Speedup: {speedup:.2f}x")
        print()

        write_csv_row(benchmark, "NumPy", time_orig, time_opt)

        # PyTorch
        # ----------------
        program_contents_torch = (
            program_contents.replace(
                "jnp.tensordot(A, B, axes=([1], [0]))", "torch.matmul(A, B)"
            )
            .replace("jnp.array", "torch.tensor")
            .replace("jnp.", "torch.")
            .replace("dot", "matmul")
            .replace("power", "pow")
        )
        if ".at[" in program_contents_torch:
            program_contents_torch = compatibalize_dot_at(program_contents_torch)
        program_contents_torch = compatibalize_transpose(program_contents_torch)
        exec(program_contents_torch, globals())
        kernel, kernel_opt, args = fetch_kernel_and_arg_ranks_from_globals()
        args = arg_ranks_to_args(arg_ranks)
        if "kernel_torch" in globals():
            kernel = kernel_torch
            del globals()["kernel_torch"]

        # Benchmark
        args = resize_arg_shapes(args)
        args = [torch.tensor(arg) if hasattr(arg, "shape") else arg for arg in args]

        print(f"Benchmark (Torch): {benchmark_file}")

        kernel = torch.compile(kernel, backend="inductor")
        kernel_opt = torch.compile(kernel_opt, backend="inductor")
        time_orig = bench(kernel, args)
        time_opt = bench(kernel_opt, args)
        speedup = time_orig / time_opt

        print(f"Original time: {time_orig:.6f} seconds")
        print(f"Optimized time: {time_opt:.6f} seconds")
        print(f"Speedup: {speedup:.2f}x")
        print("-" * 40)

        write_csv_row(benchmark, "PyTorch", time_orig, time_opt)

    cmd = ["Rscript", os.path.join(SCRIPT_DIR, "plot_perf.R")]
    subprocess.run(cmd, cwd=FLAGS.out_dir)


if __name__ == "__main__":
    app.run(main)
