# RUN: python %s | FileCheck %s

from mlir_synth.ir import *
import mlir_synth.synth as mlir_synth

ASM = """
module {
  func.func @foo(%arg0: memref<3x3xf64>, %arg1: f64, %arg2: memref<3x3xf64>, %arg3: memref<3x3xf64>, %alloc: memref<3x3xf64>) -> memref<3x3xf64> {
    memref.copy %arg3, %alloc : memref<3x3xf64> to memref<3x3xf64>
    affine.for %arg4 = 0 to 3 {
      affine.for %arg5 = 0 to 3 {
        affine.for %arg6 = 0 to 3 {
          %0 = affine.load %arg0[%arg4, %arg5] : memref<3x3xf64>
          %1 = arith.mulf %arg1, %0 : f64
          %2 = affine.load %arg2[%arg5, %arg6] : memref<3x3xf64>
          %3 = arith.mulf %1, %2 : f64
          %4 = affine.load %alloc[%arg4, %arg6] : memref<3x3xf64>
          %5 = arith.addf %4, %3 : f64
          affine.store %5, %alloc[%arg4, %arg6] : memref<3x3xf64>
        }
      }
    }
    return %alloc : memref<3x3xf64>
  }
}
"""
with Context():
  mlir_synth.register_dialects()

  m = Module.parse(ASM)

  operations = [
    "chlo.broadcast_divide",
    "chlo.broadcast_add",
    "chlo.broadcast_subtract",
    "chlo.broadcast_multiply",
    "stablehlo.dot",
    "stablehlo.reduce",
    "stablehlo.dot_general",
    "stablehlo.transpose",
    "stablehlo.select"
  ]
  options = {
    "maxNumOps": 1,
    "printValidCandidates": False,
    "printSynthesisSteps": True,
    "stopOnSolutionCandidate": True,
    "ignoreEquivalentCandidates": True,
  }
  hlo_module_strs = []
  mlir_synth.synthesize(m, operations, options, hlo_module_strs)

  assert len(hlo_module_strs) > 0

# CHECK: tensor
  print (hlo_module_strs[0])