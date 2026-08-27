#!/bin/sh
set -x
set -e

# Build the image
docker build -t stenso-artifact -f Dockerfile .

# Run synthesis
mkdir -p out
#docker run -v $(pwd)/out:/root/out --cap-add=SYS_NICE -it stenso-artifact "numactl --physcpubind=0-15,32-47 --membind=0 python stenso/eval/run_synthesis.py --with_measuring_estimator --with_branch_and_bound --with_exhaustive --with_bottom_up --timeout 10"
docker run -v $(pwd)/out:/root/out -it stenso-artifact "python stenso/eval/run_synthesis.py --with_measuring_estimator --with_branch_and_bound --with_exhaustive --with_bottom_up"

# Merge original and synthesized programs into file
docker run -v $(pwd)/out:/root/out -it stenso-artifact "python stenso/eval/merge_benchmarks.py"

# Run performance measurements
docker run -v $(pwd)/out:/root/out -it stenso-artifact "python stenso/eval/run_performance.py"