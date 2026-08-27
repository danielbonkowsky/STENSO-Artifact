func.func @fn_2(%arg0: memref<2x7xf64>, %arg1: memref<2x3xf64>, %arg2: memref<3x7xf64>) -> memref<2x7xf64> attributes {irsynth.original} {
  %cst = arith.constant 0.000000e+00 : f64
  affine.for %arg3 = 0 to 2 {
    affine.for %arg4 = 0 to 7 {
      affine.store %cst, %arg0[%arg3, %arg4] : memref<2x7xf64>
      affine.for %arg5 = 0 to 3 {
        %0 = affine.load %arg1[%arg3, %arg5] : memref<2x3xf64>
        %1 = affine.load %arg2[%arg5, %arg4] : memref<3x7xf64>
        %2 = arith.mulf %0, %1 : f64
        %3 = affine.load %arg0[%arg3, %arg4] : memref<2x7xf64>
        %4 = arith.addf %3, %2 : f64
        affine.store %4, %arg0[%arg3, %arg4] : memref<2x7xf64>
      }
    }
  }
  return %arg0 : memref<2x7xf64>
}