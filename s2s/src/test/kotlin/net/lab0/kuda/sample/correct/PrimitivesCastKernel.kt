package net.lab0.kuda.sample.correct

import net.lab0.kuda.annotation.Global
import net.lab0.kuda.annotation.Kernel

@Kernel
class PrimitivesCastKernel {
  @Global
  fun cast(
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
    bytes[0] = aByte.toByte()
    shorts[0] = aByte.toShort()
    ints[0] = aByte.toInt()
    longs[0] = aByte.toLong()
    floats[0] = aByte.toFloat()
    doubles[0] = aByte.toDouble()

    bytes[2] = aShort.toByte()
    shorts[2] = aShort.toShort()
    ints[2] = aShort.toInt()
    longs[2] = aShort.toLong()
    floats[2] = aShort.toFloat()
    doubles[2] = aShort.toDouble()

    bytes[4] = aInt.toByte()
    shorts[4] = aInt.toShort()
    ints[4] = aInt.toInt()
    longs[4] = aInt.toLong()
    floats[4] = aInt.toFloat()
    doubles[4] = aInt.toDouble()

    bytes[6] = aLong.toByte()
    shorts[6] = aLong.toShort()
    ints[6] = aLong.toInt()
    longs[6] = aLong.toLong()
    floats[6] = aLong.toFloat()
    doubles[6] = aLong.toDouble()

    bytes[8] = aFloat.toByte()
    shorts[8] = aFloat.toShort()
    ints[8] = aFloat.toInt()
    longs[8] = aFloat.toLong()
    floats[8] = aFloat.toFloat()
    doubles[8] = aFloat.toDouble()

    bytes[9] = aDouble.toByte()
    shorts[9] = aDouble.toShort()
    ints[9] = aDouble.toInt()
    longs[9] = aDouble.toLong()
    floats[9] = aDouble.toFloat()
    doubles[9] = aDouble.toDouble()

  }
}