package net.lab0.kuda

import net.lab0.kuda.sample.s1.WhileKernel
import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Test
import java.nio.file.Files

class WhileKernelTest {

  @Test
  fun `can convert while loops`() {
    val source = loadAndTranspile(WhileKernel::class)

    val reference = """
      |extern "C"
      |
      |__global__
      |void whileLoop()
      |{
      |  int idx = blockDim.x;
      |  bool b;
      |  while (idx < 10) {
      |    b = true;
      |  }
      |}
    """.trimMargin()

    assertPtxEquals(source, reference)
  }
}
