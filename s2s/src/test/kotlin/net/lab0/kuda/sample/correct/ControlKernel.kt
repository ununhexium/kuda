package net.lab0.kuda.sample.correct

import net.lab0.kuda.KudaContext
import net.lab0.kuda.annotation.Global
import net.lab0.kuda.annotation.Kernel

@Kernel
class ControlKernel : KudaContext() {
  @Global
  fun whileLoop(ints: IntArray) {
    val idx: Int = blockDim.x
    while (idx < 10) {
      ints[idx] = 1
      if (idx == 5) {
        ints[idx] = -1
      }
    }
  }
}
