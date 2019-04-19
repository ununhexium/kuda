package net.lab0.kuda.sample.correct

import net.lab0.kuda.KudaContext
import net.lab0.kuda.annotation.Global
import net.lab0.kuda.annotation.Kernel

@Kernel
class DataTypeKernel : KudaContext() {
  @UseExperimental(ExperimentalUnsignedTypes::class)
  @Global
  fun dataTypes(
      aByte: Byte,
      bytes: ByteArray,

      aShort: Short,
      shorts: ShortArray,

      aInt: Int,
      ints: IntArray,

      aLong: Long,
      longs: LongArray,

      aFloat: Float,
      floats: FloatArray,

      aDouble: Double,
      doubles: DoubleArray
  ) {
    bytes[0] = aByte

    shorts[0] = aShort

    ints[0] = aInt

    // TODO: but why? To be able to check for underflow while keeping the compiler on the JVM?!?! Isn't that handled as a BigInt internally in the compiler?
    longs[0] = aLong

    floats[0] = aFloat
    doubles[0] = aDouble
  }
}
