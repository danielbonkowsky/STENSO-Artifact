// RUN: translate-affine-to-python %s | FileCheck %s

// CHECK: def
module {
  memref.global "private" constant @__constant_2xi64 : memref<2xi64> = dense<[5, 3]> {alignment = 64 : i64}
  func.func @foo() -> memref<2xi64> {
    %0 = memref.get_global @__constant_2xi64 : memref<2xi64>
    return %0 : memref<2xi64>
  }
}
