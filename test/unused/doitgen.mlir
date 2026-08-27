module {
  func.func @foo(%arg0: tensor<3x3x3xf64> {irsynth.fnArgIdx = 1 : index}, %arg1: tensor<3x3xf64> {irsynth.fnArgIdx = 2 : index}) -> tensor<3x3x3xf64> attributes {llvm.emit_c_interface} {
    %0 = "stablehlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [0]>} : (tensor<3x3x3xf64>, tensor<3x3xf64>) -> tensor<3x3x3xf64>
    return %0 : tensor<3x3x3xf64>
  }
}