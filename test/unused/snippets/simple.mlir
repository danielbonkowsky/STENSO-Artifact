module {
  func.func @foo(%arg6 : memref<5x5xf64>) -> memref<5x5xf64> attributes { llvm.emit_c_interface} {
    %0 = memref.alloc() : memref<5x5xf64>
    return %0: memref<5x5xf64>
  }
}
