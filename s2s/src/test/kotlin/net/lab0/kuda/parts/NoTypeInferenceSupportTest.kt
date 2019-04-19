package net.lab0.kuda.parts

import net.lab0.kuda.exception.CantConvert
import net.lab0.kuda.loadAndTranspile
import net.lab0.kuda.sample.invalid.NoTypeInferenceSupport
import org.assertj.core.api.Assertions
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows

class NoTypeInferenceSupportTest {
  @Test
  fun `output error message when the variable type is not specified`() {
    val exception = assertThrows<CantConvert> {
      loadAndTranspile(NoTypeInferenceSupport::class)
    }

    Assertions.assertThat(exception.message).contains(
        "There is no type inference. As for now, you must specify the type of your left hand operand."
    )
  }
}