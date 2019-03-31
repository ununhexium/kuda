package net.lab0.kuda.sample.s1

import net.lab0.kuda.KudaContext
import net.lab0.kuda.generator.Global
import net.lab0.kuda.generator.Kernel

@Kernel
class K1 : KudaContext() {
  @Global
  fun saxpy(n: Int, a: Float, x: FloatArray, y: FloatArray) {
    val i: Int = blockIdx.x * blockDim.x + threadIdx.x
    if (i < n) y[i] = a * x[i] + y[i]
  }
}
