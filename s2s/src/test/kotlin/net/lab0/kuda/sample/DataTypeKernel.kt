package net.lab0.kuda.sample

import net.lab0.kuda.KudaContext
import net.lab0.kuda.annotation.Global
import net.lab0.kuda.annotation.Kernel

@Kernel
class DataTypeKernel : KudaContext() {
  @UseExperimental(ExperimentalUnsignedTypes::class)
  @Global
  fun dataTypes(
      aBool: Boolean,
      bools: BooleanArray,

      aByte: Byte,
      bytes: ByteArray,
      aUByte: UByte,
      ubytes: UByteArray,

      aShort: Short,
      shorts: ShortArray,
      auShort: UShort,
      ushorts: UShortArray,

      aInt: Int,
      ints: IntArray,
      aUInt: UInt,
      uints: UIntArray,

      aLong: Long,
      longs: LongArray,
      aULong: ULong,
      ulongs: ULongArray,

      aFloat: Float,
      floats: FloatArray,

      aDouble: Double,
      doubles: DoubleArray
  ) {
    bools[0] = aBool

    bytes[0] = aByte
    ubytes[0] = aUByte

    shorts[0] = aShort
    ushorts[0] = auShort

    ints[0] = aInt
    uints[0] = aUInt

    // TODO: but why? To be able to check for underflow while keeping the compiler on the JVM?!?! Isn't that handled as a BigInt internally in the compiler?
    longs[0] = aLong
    ulongs[0] = aULong

    floats[0] = aFloat
    doubles[0] = aDouble
  }
}
