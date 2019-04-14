package net.lab0.kuda

import net.lab0.kuda.sample.s1.AddConstantKernel
import org.junit.jupiter.api.Test

class AddConstantKernelTest {
  @Test
  fun `can add constant`() {
    val cuda = loadAndTranspile(AddConstantKernel::class)

    val expected =
        """
          |extern "C"
          |
          |__global__
          |void addN(int n, int * ns)
          |{
          |  int idx = blockIdx.x * blockDim.x + threadIdx.x;
          |  ns[idx] = ns[idx] + n;
          |}
        """.trimMargin()

    assertPtxEquals(cuda, expected)
  }
}