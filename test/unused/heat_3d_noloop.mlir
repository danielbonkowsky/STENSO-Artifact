module {
  func.func @kernel_heat_3d(%arg2: memref<5x5xf64>, %arg3: memref<3x3xf64>) {
    affine.for %arg6 = 1 to 4 {
      affine.for %arg7 = 1 to 4 {
        %0 = affine.load %arg2[%arg6, %arg7] : memref<5x5xf64>

        %1 = affine.load %arg2[%arg6 - 1, %arg7] : memref<5x5xf64>
        %2 = arith.addf %1, %0 : f64

        %3 = affine.load %arg2[%arg6 + 1, %arg7] : memref<5x5xf64>
        %4 = arith.addf %3, %2 : f64

        %5 = affine.load %arg2[%arg6, %arg7 - 1] : memref<5x5xf64>
        %6 = arith.addf %5, %4 : f64

        %7 = affine.load %arg2[%arg6, %arg7 + 1] : memref<5x5xf64>
        %8 = arith.addf %7, %6 : f64

        %9 = affine.load %arg2[%arg6 - 1, %arg7 - 1] : memref<5x5xf64>
        %10 = arith.addf %9, %8 : f64

        %11 = affine.load %arg2[%arg6 - 1, %arg7 + 1] : memref<5x5xf64>
        %12 = arith.addf %11, %10 : f64

        %13 = affine.load %arg2[%arg6 + 1, %arg7 - 1] : memref<5x5xf64>
        %14 = arith.addf %13, %12 : f64

        %15 = affine.load %arg2[%arg6 + 1, %arg7 + 1] : memref<5x5xf64>
        %16 = arith.addf %15, %14 : f64

        //%17 = arith.constant 9.000000e+00 : f64
        //%18 = arith.divf %16, %17 : f64

        affine.store %16, %arg3[%arg6 - 1, %arg7 - 1] : memref<3x3xf64>
      }
    }

    return
  }
}