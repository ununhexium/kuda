package net.lab0.kuda.sample.s1

import net.lab0.kuda.KudaContext
import net.lab0.kuda.annotation.Global
import net.lab0.kuda.annotation.Kernel

@Kernel
class WhileKernel : KudaContext() {
  @Global
  fun whileLoop() {
    val idx: Int = blockDim.x
    var b: Boolean
    while (idx < 10) {
      b = true
    }
  }
}
