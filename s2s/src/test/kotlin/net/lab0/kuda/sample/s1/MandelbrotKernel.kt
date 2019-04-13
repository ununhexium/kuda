package net.lab0.kuda.sample.s1

import net.lab0.kuda.KudaContext
import net.lab0.kuda.annotation.Global
import net.lab0.kuda.annotation.Kernel

@Kernel
class MandelbrotKernel : KudaContext() {
  @Global
  fun mandelbrot(reals: DoubleArray, imags: DoubleArray, iterationLimit: Long, iterations: LongArray) {
    val idx: Int = blockIdx.x * blockDim.x + threadIdx.x

    val real: Double = reals[idx]
    val imag: Double = imags[idx]
    var real1: Double = real
    var imag1: Double = imag
    var real2: Double
    var imag2: Double

    var iter: Long = 0
    while (iter < iterationLimit && real1 <= 2.0 && real1 >= -2.0 && imag1 <= 2.0 && imag1 >= -2.0) {
      real2 = real1 * real1 - imag1 * imag1 + real
      imag2 = 2.0 * real1 * imag1 + imag

      real1 = real2
      imag1 = imag2

      iter++
    }

    iterations[idx] = iter
  }
}
