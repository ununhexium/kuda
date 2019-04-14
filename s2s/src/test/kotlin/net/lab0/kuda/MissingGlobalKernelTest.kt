package net.lab0.kuda

import net.lab0.kuda.exception.CantConvert
import net.lab0.kuda.sample.s1.MissingGlobalKernel
import org.assertj.core.api.Assertions
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows

class MissingGlobalKernelTest {

  @Test
  fun `output a meaningful error message when the @Global annotation is missing`() {
    val exception = assertThrows<CantConvert> {
      loadAndTranspile(MissingGlobalKernel::class)
    }

    Assertions.assertThat(exception.message).contains(
        "There is no @Global function in the class MissingGlobalKernel."
    )
  }

}