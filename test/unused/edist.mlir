// def loop_edist(X: np.ndarray, Y: np.ndarray):
//     dists = np.zeros((X.shape[0], Y.shape[0]))
//     
//     for i in range(X.shape[0]):
//         for j in range(Y.shape[0]):
//             acc = 0
//             for k in range(X.shape[1]):
//                 acc += (X[i, k] - Y[j, k]) ** 2
//             dists[i, j] = math.sqrt(acc)
// 
//     return dists

module {
  func.func @edist(%arg0: memref<3x5xf64>, %arg1: memref<3x5xf64>, %arg2: memref<3x3xf64>) {
    %cst = arith.constant 0.000000e+00 : f64
    affine.for %arg3 = 0 to 3 {
      affine.for %arg4 = 0 to 3 {
        affine.store %cst, %arg2[%arg3, %arg4] : memref<3x3xf64>
      }
    }
    %alloca = memref.alloca() : memref<f64>
    affine.for %arg3 = 0 to 3 {
      affine.for %arg4 = 0 to 3 {
        affine.store %cst, %alloca[] : memref<f64>
        affine.for %arg5 = 0 to 5 {
          %2 = affine.load %arg0[%arg3, %arg5] : memref<3x5xf64>
          %3 = affine.load %arg1[%arg4, %arg5] : memref<3x5xf64>
          %4 = arith.subf %2, %3 : f64
          %5 = arith.mulf %4, %4 : f64
          %6 = affine.load %alloca[] : memref<f64>
          %7 = arith.addf %6, %5 : f64
          affine.store %7, %alloca[] : memref<f64>
        }
        %0 = affine.load %alloca[] : memref<f64>
        %1 = math.sqrt %0 : f64
        affine.store %1, %arg2[%arg3, %arg4] : memref<3x3xf64>
      }
    }
    return
  }
}