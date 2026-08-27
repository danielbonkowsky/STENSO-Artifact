module {
  func.func @kernel_3mm(%arg5: memref<5x5xf64>) attributes {changed_sizes = "1000:3,900:3,800:3,1200:3,1100:3", llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    affine.for %arg12 = 0 to 5 {
      affine.for %arg13 = 0 to 5 {
        affine.for %arg14 = 0 to 5 {
          %3 = affine.load %arg5[%arg12, %arg13] : memref<5x5xf64>
          %4 = arith.addf %cst, %3 : f64
          affine.store %4, %arg5[%arg12, %arg13] : memref<5x5xf64>
        }
      }
    }
    return
  }
}

