# RUN: python %s | FileCheck %s

from mlir_synth.ir import *
import mlir_synth.synth as mlir_synth

ASM = """
module {
  func.func @foo(%arg0: tensor<3x3xf64> {irsynth.fnArgIdx = 0 : index}) -> tensor<3x3xf64> attributes {llvm.emit_c_interface} {
    %0 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %1 = chlo.broadcast_add %arg0, %0 : (tensor<3x3xf64>, tensor<f64>) -> tensor<3x3xf64>
    return %1 : tensor<3x3xf64>
  }
}
"""

with Context():
  mlir_synth.register_dialects()

  m = Module.parse(ASM)
  print(m)

# CHECK: memref
  mlir_synth.lower_chlo_to_affine(m)
  print(m)

# CHECK: def
  print(mlir_synth.emit_python(m))