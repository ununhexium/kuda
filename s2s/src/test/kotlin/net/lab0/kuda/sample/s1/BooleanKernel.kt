package net.lab0.kuda.sample.s1

import net.lab0.kuda.annotation.Global
import net.lab0.kuda.annotation.Kernel

@Kernel
class BooleanKernel {
  @Global
  fun booleanStuff() {
    var t: Boolean = true
    var f: Boolean = false
  }
}