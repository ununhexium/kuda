package net.lab0.kuda.parts

import net.lab0.kuda.annotation.Global
import net.lab0.kuda.annotation.Kernel
import net.lab0.kuda.assertPtxEquals
import net.lab0.kuda.loadAndTranspile
import net.lab0.kuda.sample.correct.ArrayDeclarationKernel

// TODO: test and implement
@Kernel
class ArrayDeclarationKernelTest {
  @Global
  fun `declare arrays and initialise them`() {
    /**
     *
    fun declareArrays() {
    val floats: FloatArray = FloatArray(116)
    val initFloats: FloatArray = FloatArray(116) { 1.0f }

    val doubles: DoubleArray = DoubleArray(116)
    val initDoubles: DoubleArray = DoubleArray(116) { 1.0 }
    }
     */
    assertPtxEquals(
        """
          |extern "C"
          |
          |__global__ void declareArrays() {
          |  float [] floats = float[116];
          |  float [] initFloats = float[116];
          |  for(int initFloats__idx; initFloats__idx < 116; initFloats__idx++) {
          |    initFloats[initFloats__idx] = 1.0f
          |  }
          |}
        """.trimMargin(),
        loadAndTranspile(ArrayDeclarationKernel::class)
    )
  }
}
