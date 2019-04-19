package net.lab0.kuda.parts

import net.lab0.kuda.StandardKernelTest
import net.lab0.kuda.sample.correct.MandelbrotKernel

class MandelbrotKernelTest : StandardKernelTest(MandelbrotKernel::class) {
  override fun getEquivalentCKernel() =
      """
        |extern "C"
        |
        |__global__ void mandelbrot(double * reals, double * imags, long iterationLimit, long * iterations) {
        |  int idx = blockIdx.x * blockDim.x + threadIdx.x;
        |
        |  double real = reals[idx];
        |  double imag = imags[idx];
        |
        |  double real1 = 0.0;
        |  double imag1 = 0.0;
        |  double real2;
        |  double imag2;
        |
        |  long iter = 0;
        |
        |  while (iter < iterationLimit && real1 <= 2.0 && real1 >= -2.0 && imag1 <= 2.0 && imag1 >= -2.0) {
        |    real2 = real1 * real1 - imag1 * imag1 + real;
        |    imag2 = 2.0 * real1 * imag1 + imag;
        |
        |    real1 = real2;
        |    imag1 = imag2;
        |
        |    iter++;
        |  }
        |
        |  iterations[idx] = iter;
        |}
      """.trimMargin()
}
