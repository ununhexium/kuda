package net.lab0.kuda.sample.s1

import net.lab0.kuda.KudaContext
import net.lab0.kuda.annotation.Global
import net.lab0.kuda.annotation.Kernel

@Kernel
class K0 : KudaContext() {
  @Global
  fun count(ints: IntArray) {
    val idx: Int = blockIdx.x * blockDim.x + threadIdx.x
    ints[idx] = idx
  }
}
