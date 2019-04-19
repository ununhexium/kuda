package net.lab0.kuda.sample

import net.lab0.kuda.KudaContext
import net.lab0.kuda.annotation.Global
import net.lab0.kuda.annotation.Kernel

@Kernel
class ControlKernel : KudaContext() {
  @Global
  fun whileLoop(bools: BooleanArray) {
    val idx: Int = blockDim.x
    while (idx < 10) {
      bools[idx] = true
      if (idx == 5) {
        bools[idx] = false
      }
    }
  }
}
