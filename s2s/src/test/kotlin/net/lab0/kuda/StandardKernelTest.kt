package net.lab0.kuda

import net.lab0.kuda.wrapper.CallWrapperGenerator
import org.junit.jupiter.api.Test
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import java.io.StringWriter
import kotlin.reflect.KClass

abstract class StandardKernelTest(val kClass: KClass<*>) {
  companion object
  {
      private val log: Logger by lazy { LoggerFactory.getLogger(this::class.java.name) }
  }

  @Test
  fun testTranspilation() {
    assertPtxEquals(
        loadAndTranspile(kClass),
        getEquivalentCKernel()
    )
  }

  @Test
  fun `can generate a wrapper`() {
    val outputPackage = "net.lab0.kuda.example"
    val gen = CallWrapperGenerator(loadSource(kClass), outputPackage)
    val wrapper = gen.callWrapperFile
    val buffer = StringWriter()
    wrapper.writeTo(buffer)

    log.info(
        """
          |Generated a wrapper for $kClass:
          |$buffer
          |
        """.trimMargin())
  }

  abstract fun getEquivalentCKernel(): String
}
