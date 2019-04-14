package net.lab0.kuda.sample.s1

import net.lab0.kuda.KudaContext
import net.lab0.kuda.annotation.Global
import net.lab0.kuda.annotation.Kernel

@Kernel
class DoNothing : KudaContext() {
  @Suppress("unused")
  @Global
  fun doNothing() {
    // that's it folks!
  }
}
