package net.lab0.kuda

import net.lab0.kuda.sample.ControlKernel
import org.junit.jupiter.api.Test

class ControlKernelTest {

  @Test
  fun `can convert while loops`() {
    val source = loadAndTranspile(ControlKernel::class)

    val reference = """
      |extern "C"
      |
      |__global__
      |void whileLoop(bool * bools)
      |{
      |  int idx = blockDim.x;
      |  while (idx < 10) {
      |    bools[idx] = true;
      |    if(5 == idx) {
      |      bools[idx] = false;
      |    }
      |  }
      |}
    """.trimMargin()

    assertPtxEquals(source, reference)
  }
}
