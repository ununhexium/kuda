package net.lab0.kuda.wrapper

import com.squareup.kotlinpoet.KModifier
import com.squareup.kotlinpoet.asTypeName
import net.lab0.kuda.KernelParameters
import net.lab0.kuda.KudaContext
import net.lab0.kuda.loadSource
import net.lab0.kuda.named
import net.lab0.kuda.sample.s1.DoNothingKernel
import net.lab0.kuda.sample.s1.SaxpyKernel
import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Test

internal class CallWrapperGeneratorTest {

  @Test
  fun `generate a wrapper class for DoNothing kernel`() {
    val source = loadSource(DoNothingKernel::class)
    val generator = CallWrapperGenerator(source, "foo.pack")
    val classname = generator.className

    assertThat(classname.simpleName).isEqualTo("DoNothingKernelWrapper")

    val wrapper = generator.callWrapper

    assertThat(wrapper.superclass).isEqualTo(KudaContext::class.asTypeName())

    val cudaResourceName = wrapper.propertySpecs.named("cudaResourceName")
    assertThat(cudaResourceName.initializer).isNotNull
    assertThat(cudaResourceName.initializer.toString()).isEqualTo(
        """
          "foo/pack/DoNothingKernel.kt.cu"
        """.trimIndent()
    )

    val invoke = wrapper.funSpecs.named("invoke")
    assertThat(invoke.modifiers).contains(KModifier.OPERATOR)
    assertThat(invoke.parameters).hasSize(1)
    assertThat(invoke.returnType).isEqualTo(Unit::class.asTypeName())

    val kernelParams = invoke.parameters.first()
    assertThat(kernelParams.name).isEqualTo("kernelParameters")
    assertThat(kernelParams.type).isEqualTo(KernelParameters::class.asTypeName())
  }

  @Test
  fun `generate a wrapper class for saxpy`() {
    val source = loadSource(SaxpyKernel::class)
    val wrapper = CallWrapperGenerator(source, "foo.pack").callWrapper

    val invoke = wrapper.funSpecs.named("invoke")
    // when there is only 1 return parameter, we can return it directly
    assertThat(invoke.returnType).isEqualTo(FloatArray::class.asTypeName())

    val n = invoke.parameters.named("n")
    assertThat(n.type).isEqualTo(Int::class.asTypeName())

    val a = invoke.parameters.named("a")
    assertThat(a.type).isEqualTo(Float::class.asTypeName())

    val x = invoke.parameters.named("x")
    assertThat(x.type).isEqualTo(FloatArray::class.asTypeName())

    val y = invoke.parameters.named("y")
    assertThat(x.type).isEqualTo(FloatArray::class.asTypeName())

    // Possible improvement: params marked as @Return could ask for a size instead of an array
  }

}
