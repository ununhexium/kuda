package net.lab0.kuda.sample.s1

import net.lab0.kuda.KudaContext
import net.lab0.kuda.annotation.Global
import net.lab0.kuda.annotation.Kernel

@Kernel
class K4 : KudaContext() {

  @Global
  fun addN(n: Int, ns: IntArray) {
    val idx: Int = blockIdx.x * blockDim.x + threadIdx.x
    ns[idx] = ns[idx] + n
  }

}