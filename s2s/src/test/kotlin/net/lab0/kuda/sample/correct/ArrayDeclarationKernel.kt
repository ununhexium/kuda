package net.lab0.kuda.sample.correct

import net.lab0.kuda.annotation.Global
import net.lab0.kuda.annotation.Kernel

@Kernel
class ArrayDeclarationKernel {
  @Global
  fun declareArrays() {
    val floats: FloatArray = FloatArray(116)
    val initFloats: FloatArray = FloatArray(116) { 1.0f }

    val doubles: DoubleArray = DoubleArray(116)
    val initDoubles: DoubleArray = DoubleArray(116) { 1.0 }
  }
}