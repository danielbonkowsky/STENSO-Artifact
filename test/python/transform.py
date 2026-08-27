# RUN: python %s | FileCheck %s

from mlir_synth.ir import *
from mlir_synth.passmanager import *
import mlir_synth.synth as mlir_synth

ASM = """
module {
  func.func @foo(%arg0: memref<1000x2000xf64>, %arg1: memref<1000x2000xf64>) -> memref<1000x2000xf64> {
    %res = memref.alloc() : memref<1000x2000xf64>

    affine.for %i = 0 to 1000 {
      affine.for %j = 0 to 2000 {
        %0 = affine.load %arg0[%i, %j] : memref<1000x2000xf64>
        %1 = affine.load %arg1[%i, %j] : memref<1000x2000xf64>
        %2 = arith.addf %0, %1 : f64
        affine.store %2, %res[%i, %j] : memref<1000x2000xf64>
      }
    }

    return %res : memref<1000x2000xf64>
  }
}
"""

with Context():
  mlir_synth.register_dialects()
  mlir_synth.register_passes()

  m = Module.parse(ASM)

# CHECK: memref<1000x2000xf64>
  mlir_synth.lower_chlo_to_affine(m)
  print(m)

  pm = PassManager.parse("change-sizes{sizes=Uniform}")
  pm.run(m)

# CHECK: memref<3x3xf64>
  print(m)