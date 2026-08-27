// RUN: python %mlirsynth_src_root%/sandbox/sym_solve/main.py %s | FileCheck %s

func.func @foo(%arg0: memref<3x7xf64>, %arg1: f64, %arg2: memref<7x5xf64>, %arg3: memref<3x5xf64> {irSynth.lastStoredMemref}) -> memref<3x5xf64> attributes {irsynth.original} {
  %alloc = memref.alloc() : memref<3x5xf64>
  memref.copy %arg3, %alloc : memref<3x5xf64> to memref<3x5xf64>
  affine.for %arg4 = 0 to 3 {
    affine.for %arg5 = 0 to 7 {
      affine.for %arg6 = 0 to 5 {
        %0 = affine.load %arg0[%arg4, %arg5] : memref<3x7xf64>
        %1 = arith.mulf %arg1, %0 : f64
        %2 = affine.load %arg2[%arg5, %arg6] : memref<7x5xf64>
        %3 = arith.mulf %1, %2 : f64
        %4 = affine.load %alloc[%arg4, %arg6] : memref<3x5xf64>
        %5 = arith.addf %4, %3 : f64
        affine.store %5, %alloc[%arg4, %arg6] : memref<3x5xf64>
      }
    }
  }
  return %alloc : memref<3x5xf64>
}