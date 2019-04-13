package net.lab0.kuda

import net.lab0.kuda.sample.s1.BooleanKernel
import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Test

class BooleanKernelTest {
  @Test
  fun `boolean assignment`() {
    assertThat(
        loadAndTranspile(BooleanKernel::class)
    ).isEqualTo(
        """
          |extern "C"
          |
          |__global__
          |void booleanStuff()
          |{
          |  bool t = true;
          |  bool f = false;
          |}
        """.trimMargin()
    )
  }
}