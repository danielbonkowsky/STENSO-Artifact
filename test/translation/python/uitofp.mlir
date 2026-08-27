// RUN: translate-affine-to-python %s | FileCheck %s

// CHECK: return
module @jit_foo {
  func.func public @main(%arg0: memref<f64>, %arg1: memref<f64>) -> memref<f64> {
    %cst = arith.constant 2.000000e+00 : f64
    %0 = memref.load %arg0[] : memref<f64>
    %1 = memref.load %arg1[] : memref<f64>
    %2 = arith.cmpf olt, %cst, %0 : f64
    %3 = arith.uitofp %2 : i1 to f64
    %4 = arith.addf %3, %1 : f64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<f64>
    memref.store %4, %alloc[] : memref<f64>
    return %alloc : memref<f64>
  }
}