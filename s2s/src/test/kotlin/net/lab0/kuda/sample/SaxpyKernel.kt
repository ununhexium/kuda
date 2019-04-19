package net.lab0.kuda.sample

import net.lab0.kuda.KudaContext
import net.lab0.kuda.annotation.Global
import net.lab0.kuda.annotation.Kernel
import net.lab0.kuda.annotation.Return

@Kernel
class SaxpyKernel : KudaContext() {
  @Global
  fun saxpy(n: Int, a: Float, x: FloatArray, @Return y: FloatArray) {
    val i: Int = blockIdx.x * blockDim.x + threadIdx.x
    if (i < n) y[i] = a * x[i] + y[i]
  }
}
