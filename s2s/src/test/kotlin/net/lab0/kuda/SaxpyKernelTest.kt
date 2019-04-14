package net.lab0.kuda

import net.lab0.kuda.sample.s1.SaxpyKernel
import org.junit.jupiter.api.Test

class SaxpyKernelTest  {
  @Test
  fun `can convert a simple kotlin kernel to a C kernel`() {
    val cuda = loadAndTranspile(SaxpyKernel::class)

    val reference =
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

    assertPtxEquals(cuda, reference)
  }
}