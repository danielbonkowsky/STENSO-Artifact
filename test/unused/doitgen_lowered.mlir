module {
  func.func @foo(%arg0: memref<3x3x3xf64> {irsynth.fnArgIdx = 1 : index}, %arg1: memref<3x3xf64> {irsynth.fnArgIdx = 2 : index}) -> memref<3x3x3xf64> attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c9 = arith.constant 9 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %cst = arith.constant 0.000000e+00 : f64
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1], [2]] : memref<3x3x3xf64> into memref<9x3xf64>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<9x3xf64>
    scf.for %arg2 = %c0 to %c9 step %c1 {
      scf.for %arg3 = %c0 to %c3 step %c1 {
        memref.store %cst, %alloc[%arg2, %arg3] : memref<9x3xf64>
      }
    }
    scf.for %arg2 = %c0 to %c9 step %c1 {
      scf.for %arg3 = %c0 to %c3 step %c1 {
        scf.for %arg4 = %c0 to %c3 step %c1 {
          %0 = memref.load %collapse_shape[%arg2, %arg4] : memref<9x3xf64>
          %1 = memref.load %arg1[%arg4, %arg3] : memref<3x3xf64>
          %2 = memref.load %alloc[%arg2, %arg3] : memref<9x3xf64>
          %3 = arith.mulf %0, %1 : f64
          %4 = arith.addf %2, %3 : f64
          memref.store %4, %alloc[%arg2, %arg3] : memref<9x3xf64>
        }
      }
    }
    %expand_shape = memref.expand_shape %alloc [[0, 1], [2]] : memref<9x3xf64> into memref<3x3x3xf64>
    return %expand_shape : memref<3x3x3xf64>
  }
}