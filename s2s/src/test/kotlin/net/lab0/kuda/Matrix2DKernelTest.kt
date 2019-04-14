package net.lab0.kuda

import net.lab0.kuda.sample.s1.Matrix2DKernel
import org.junit.jupiter.api.Test

internal class Matrix2DKernelTest {
  @Test
  fun `can convert a kernel using 2D arrays`() {
    val cuda = loadAndTranspile(Matrix2DKernel::class)

    val expected =
        """
          |extern "C"
          |
          |__global__
          |void matrixAdd(float * * A, float * * B, float * * C)
          |{
          |  int i = threadIdx.x;
          |  int j = threadIdx.y;
          |  C[i][j] = A[i][j] + B[i][j];
          |}
        """.trimMargin()

    assertPtxEquals(cuda, expected)
  }
}