package net.lab0.kuda.parts

import net.lab0.kuda.StandardKernelTest
import net.lab0.kuda.sample.correct.SaxpyKernel

class SaxpyKernelTest : StandardKernelTest(SaxpyKernel::class) {
  override fun getEquivalentCKernel() =
      """
        |extern "C"
        |
        |__global__
        |void saxpy(int n, float a, float * x, float * y)
        |{
        |  int i = blockIdx.x * blockDim.x + threadIdx.x;
        |  if (i < n) {
        |    y[i] = a * x[i] + y[i];
        |  };
        |}
      """.trimMargin()
}