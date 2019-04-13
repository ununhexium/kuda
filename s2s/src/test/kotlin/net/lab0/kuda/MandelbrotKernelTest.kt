package net.lab0.kuda

import net.lab0.kuda.sample.s1.MandelbrotKernel
import org.junit.jupiter.api.Test

class MandelbrotKernelTest {
  @Test
  fun `compute fancy fractal`() {
    val expected = """
      |extern "C"
      |
      |__global__
      |void mandelbrot(double *reals, double *imags, long iterationLimit, long *iterations) {
      |  int idx = blockIdx.x * blockDim.x + threadIdx.x;
      |
      |  double real = reals[idx];
      |  double imag = imags[idx];
      |  double real1 = real;
      |  double imag1 = imag;
      |  double real2;
      |  double imag2;
      |
      |  long iter = 0;
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
      |
    """.trimMargin()

    val source = loadAndTranspile(MandelbrotKernel::class)

    assertPtxEquals(source, expected)
  }
}
