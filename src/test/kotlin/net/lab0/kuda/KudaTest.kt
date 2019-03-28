package net.lab0.kuda

import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Test
import test.sample.t1.KernelAnnotatedClass1
import test.sample.t2.GlobalFoo


internal class KudaTest {

  @Test
  fun `Can detect classes with Kernel annotation`() {
    assertThat(
        Kuda().scan("test.sample.t1")
    ).containsExactly(
        KernelAnnotatedClass1::class.java
    )
  }

  @Test
  fun `can generate most basic kernel`() {
    assertThat(
        Kuda().generateC(GlobalFoo::class.java)
    ).isEqualToNormalizingWhitespace(
        """
          |__global__
          |void foo() {
          |
          |}
          |
        """.trimMargin()
    )
  }

  @Test
  fun `can compile the most basic kernel`() {

  }
}

