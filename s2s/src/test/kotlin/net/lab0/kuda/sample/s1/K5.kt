package net.lab0.kuda.sample.s1

import net.lab0.kuda.KudaContext
import net.lab0.kuda.annotation.Global
import net.lab0.kuda.annotation.Kernel

@Kernel
class K5 : KudaContext() {

  @Global
  fun mustGiveTyper(n: Int, ns: IntArray) {
    val n = blockDim.x
  }

}