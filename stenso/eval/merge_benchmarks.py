import os
import sys
import re

from absl import app
from absl import flags


FLAGS = flags.FLAGS
flags.DEFINE_string("benchmark_dir", "stenso/eval/benchmarks", "Input directory with benchmark files")
flags.DEFINE_string("optimized_dir", "out/benchmarks_synthesized", "Input directory with optimized files")
flags.DEFINE_string("out_dir", "out/benchmarks_merged", "Out directory for concatenated files")


def main(argv):
    print(f"--- Starting Merging ---")
    print(f"Benchmarks Dir: {FLAGS.benchmark_dir}")
    print(f"Optimized Dir: {FLAGS.optimized_dir}")
    print(f"out Dir: {FLAGS.out_dir}")

    # Ensure the out directory exists
    try:
        os.makedirs(FLAGS.out_dir, exist_ok=True)
        print(f"Successfully ensured out directory '{FLAGS.out_dir}' exists.")
    except OSError as e:
        print(f"Error creating out directory '{FLAGS.out_dir}': {e}")
        return

    processed_count = 0
    skipped_count = 0

    # Iterate through files in the benchmarks directory
    for filename in os.listdir(FLAGS.benchmark_dir):
        # Only process files, skip directories
        if not os.path.isfile(os.path.join(FLAGS.benchmark_dir, filename)):
            continue

        benchmark_filepath = os.path.join(FLAGS.benchmark_dir, filename)
        optimized_filepath = os.path.join(FLAGS.optimized_dir, filename)
        out_filepath = os.path.join(FLAGS.out_dir, filename)

        # Read the contents of both files
        with open(benchmark_filepath, "r") as f:
            benchmark_content = f.read()

        optimized_content = ""
        # Check if the corresponding file exists in the optimized directory
        if os.path.exists(optimized_filepath):
            with open(optimized_filepath, "r") as f:
                optimized_content = f.read()
            optimized_content = optimized_content.replace("fn_0", "kernel_stenso")
        else:
            print(
                f"  [INFO] Optimized file for '{filename}' not found in '{FLAGS.optimized_dir}'."
            )
            # Take as optimized content the original kernel function. And call it "kernel_stenso"
            match = re.search(
                r"def\s+kernel\s*\(.*?\):\s*(?:.|\n)*?(?=^def\s+|\Z)",
                benchmark_content,
                re.MULTILINE,
            )
            optimized_content = match.group(0).replace(
                "def kernel", "def kernel_stenso"
            )
            optimized_content = (
                "# [INFO] No kernel synthesized. Fall back to original kernel function\n"
                + optimized_content
            )

        # Concatenate the contents
        concatenated_content = benchmark_content + "\n\n" + optimized_content

        # Save the concatenation to the out file
        with open(out_filepath, "w") as f:
            f.write(concatenated_content)

        print(
            f"  [SUCCESS] Processed and merged '{filename}'. out saved to '{out_filepath}'."
        )
        processed_count += 1

    print(f"\n--- Processing Complete ---")
    print(f"Total files processed and merged: {processed_count}")


if __name__ == "__main__":
    app.run(main)
