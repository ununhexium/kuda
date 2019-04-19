package net.lab0.kuda.sample

import net.lab0.kuda.annotation.Global
import net.lab0.kuda.annotation.Kernel

@Kernel
class BooleanKernel {
  @Global
  fun booleanStuff(data:BooleanArray) {
    val t: Boolean = true
    val f: Boolean = false

    data[0] = t
    data[1] = f
  }
}