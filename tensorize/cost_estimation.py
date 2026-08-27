import json
import time

import stopit
import numpy as np
from common_helpers import *
from mlir_helpers import *


class CostEstimator:
    def get_cost(self, sketch):
        raise NotImplementedError


class FlopsCostEstimator(CostEstimator):
    """
    Cost estimator based on JAX FLOPS cost model
    """

    def get_cost(self, sketch):
        return sketch.stub.cost


class MeasuredCostEstimator(CostEstimator):
    """
    Cost Estimator based on real measurements from a file
    """

    def __init__(self, filename):
        self.ir_to_time = {}

        with open(filename) as f:
            data = json.load(f)

            # Gather all times
            ir_to_times = {}
            for item in data:
                ir = item["ir"]
                if ir not in ir_to_times:
                    ir_to_times[ir] = []
                ir_to_times[ir].append(item["time"])

            # Compute medians and add to database
            for ir, times in ir_to_times.items():
                self.ir_to_time[ir] = sorted(times)[len(times) // 2]
            print("Measured costs loaded from: %s" % filename)

    def get_cost(self, sketch):
        if sketch.stub.ir in self.ir_to_time:
            return self.ir_to_time[sketch.stub.ir]
        else:
            print_yellow("WARNING: No cost found. Assuming inf. Program:")
            print(sketch.stub.original)
            return float("inf")


class MeasuringCostEstimator(CostEstimator):
    """
    Cost Estimator based on real measurements
    """

    def __init__(self):
        self.sketch_to_cost = {}

    def _benchmark(self, func, args):
        # Warm up
        for _ in range(10):
            func(*args)
        # Measure
        values = []
        for _ in range(100):
            start = time.perf_counter()
            func(*args)
            end = time.perf_counter()
            values.append(end - start)
        return np.median(values)

    def _measure_sketch(self, sketch):
        prog = sketch.stub.original
        ir = sketch.stub.ir

        # Run the function
        # - Generate random inputs
        arg_shapes = get_arg_shapes_from_module_str(ir)
        N = 100 if all(len(s) <= 2 for s in arg_shapes) else 10
        random_inputs = generate_random_inputs(arg_shapes, N)

        # - Add to context
        prog = prog.replace("jnp", "np").replace("3", str(N))
        exec(prog, globals())

        # - Benchmark
        try:
            with stopit.ThreadingTimeout(1, swallow_exc=False) as timeout_ctx:
                t = self._benchmark(foo, random_inputs)
                return t
        except (ValueError, stopit.utils.TimeoutException, TypeError) as e:
            return None

    def get_cost(self, sketch):
        if sketch.stub.ir in self.sketch_to_cost:
            return self.sketch_to_cost[sketch]
        else:
            t = self._measure_sketch(sketch)
            if t is not None:
                self.sketch_to_cost[sketch] = t
                return t
            else:
                print_yellow("WARNING: Couldn't measure sketch.")
                return float("inf")
