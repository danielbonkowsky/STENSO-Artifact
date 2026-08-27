import json
import os
import subprocess

import tqdm
from absl import app
from absl import flags
from timeit import default_timer as timer


FLAGS = flags.FLAGS
flags.DEFINE_string("out_dir", "out", "Output dir")
flags.DEFINE_integer("timeout", 10, "Timeout for each benchmark in minutes")

flags.DEFINE_bool("with_branch_and_bound", False, "With branch and bound search")
flags.DEFINE_bool("with_exhaustive", False, "With exhaustive search")
flags.DEFINE_bool("with_bottom_up", False, "With bottom up search")

flags.DEFINE_bool("with_flops_estimator", False, "With FLOPS cost estimator")
flags.DEFINE_bool("with_measuring_estimator", False, "With measuring cost estimator")

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "..")
SYNTHESIZER_PROGRAM = os.path.join(ROOT_DIR, "main.py")
SYNTHESIZER_BOTTOMUP_PROGRAM = os.path.join(ROOT_DIR, "main_bottom_up.py")

BENCH_DIR = os.path.join(ROOT_DIR, "eval/benchmarks") + "/"
BENCHMARKS = [os.path.join(BENCH_DIR, b) for b in os.listdir(BENCH_DIR)]


def run_program(cmd, stdin=None):
    start = timer()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, input=stdin, stderr=subprocess.PIPE)
    end = timer()
    return p.stdout.decode("utf-8"), end - start, p.returncode


def main(argv):
    STATS_OUT_FILE = os.path.join(FLAGS.out_dir, "stats.csv")
    if not os.path.exists(FLAGS.out_dir):
        os.makedirs(FLAGS.out_dir)

    synth_out_dir = os.path.join(FLAGS.out_dir, "benchmarks_synthesized")
    if not os.path.exists(synth_out_dir):
        os.makedirs(synth_out_dir)

    # Create configs
    search_configs = []
    if FLAGS.with_branch_and_bound:
        search_configs.append("stenso_branch_and_bound")
    if FLAGS.with_exhaustive:
        search_configs.append("stenso_exhaustive")
    if FLAGS.with_bottom_up:
        search_configs.append("bottom_up")

    cost_estimator_configs = []
    if FLAGS.with_measuring_estimator:
        cost_estimator_configs.append("measuring")
    if FLAGS.with_flops_estimator:
        cost_estimator_configs.append("flops")

    # Run experiments
    for search_config in search_configs:
        for filename in tqdm.tqdm(BENCHMARKS):
            for cost_estimator in cost_estimator_configs:
                with_save = search_config == "stenso_branch_and_bound" and cost_estimator == "measuring"

                cmd = ["timeout", str(FLAGS.timeout * 60)]
                if search_config.startswith("stenso"):
                    cmd += [
                        "python",
                        SYNTHESIZER_PROGRAM,
                        "--program",
                        filename,
                        "--cost_estimator",
                        cost_estimator,
                    ]
                    if with_save:
                        cmd += ["--synth_out", os.path.join(synth_out_dir, filename.split("/")[-1])]

                    if search_config == "stenso_exhaustive":
                        cmd += ["--no_branch_and_bound"]
                else:
                    cmd += [
                        "python",
                        SYNTHESIZER_BOTTOMUP_PROGRAM,
                        "--program",
                        filename,
                    ]

                print(filename.replace(BENCH_DIR, ""), end="\t")
                out, runtime, exitcode = run_program(cmd)
                with open("/tmp/stenso.log", "a") as f:
                    f.write(out)

                # Print status code
                if exitcode == 0:
                    print("\033[92m" + "OK" + "\033[0m", end="")
                else:
                    print("\033[91m" + "Fail" + "\033[0m", end="")
                print("\t", end="")

                if exitcode != 0:
                    print()
                else:
                    statsStr = out.split("JSON: ")[1].split("\n")[0]
                    stats = json.loads(statsStr)

                    # Create result file if it does not exist
                    stats_header = stats.keys()
                    if not os.path.exists(STATS_OUT_FILE):
                        with open(STATS_OUT_FILE, "w") as f:
                            f.write(
                                "benchmark,runtime,search_config,cost_estimator,"
                                + ",".join(stats_header)
                                + "\n"
                            )

                    # Write to results file
                    with open(STATS_OUT_FILE, "a") as f:
                        bench_name = filename.split("/")[-1].split(".")[0]
                        statsVals = ",".join([str(stats[key]) for key in stats_header])
                        f.write(
                            f"{bench_name},{runtime},{search_config},{cost_estimator},{statsVals}\n"
                        )

    cmd = ["Rscript", os.path.join(SCRIPT_DIR, "plot_synth.R")]
    subprocess.run(cmd, cwd=FLAGS.out_dir)


if __name__ == "__main__":
    app.run(main)
