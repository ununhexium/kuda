package net.lab0.kuda.sample

import net.lab0.kuda.annotation.Global
import net.lab0.kuda.annotation.Kernel

@Kernel
class PrimitivesCastKernel {
  @Global
  fun cast(
      aByte: Byte,
      bytes: ByteArray,
      aUByte: UByte,
      ubytes: UByteArray,

      aShort: Short,
      shorts: ShortArray,
      aUShort: UShort,
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
    bytes[0] = aByte.toByte()
    ubytes[0] = aByte.toUByte()
    shorts[0] = aByte.toShort()
    ushorts[0] = aByte.toUShort()
    ints[0] = aByte.toInt()
    uints[0] = aByte.toUInt()
    longs[0] = aByte.toLong()
    ulongs[0] = aByte.toULong()
    floats[0] = aByte.toFloat()
    doubles[0] = aByte.toDouble()

    bytes[1] = aUByte.toByte()
    ubytes[1] = aUByte.toUByte()
    shorts[1] = aUByte.toShort()
    ushorts[1] = aUByte.toUShort()
    ints[1] = aUByte.toInt()
    uints[1] = aUByte.toUInt()
    longs[1] = aUByte.toLong()
    ulongs[1] = aUByte.toULong()

    bytes[2] = aShort.toByte()
    ubytes[2] = aShort.toUByte()
    shorts[2] = aShort.toShort()
    ushorts[2] = aShort.toUShort()
    ints[2] = aShort.toInt()
    uints[2] = aShort.toUInt()
    longs[2] = aShort.toLong()
    ulongs[2] = aShort.toULong()
    floats[2] = aShort.toFloat()
    doubles[2] = aShort.toDouble()

    bytes[3] = aUShort.toByte()
    ubytes[3] = aUShort.toUByte()
    shorts[3] = aUShort.toShort()
    ushorts[3] = aUShort.toUShort()
    ints[3] = aUShort.toInt()
    uints[3] = aUShort.toUInt()
    longs[3] = aUShort.toLong()
    ulongs[3] = aUShort.toULong()

    bytes[4] = aInt.toByte()
    ubytes[4] = aInt.toUByte()
    shorts[4] = aInt.toShort()
    ushorts[4] = aInt.toUShort()
    ints[4] = aInt.toInt()
    uints[4] = aInt.toUInt()
    longs[4] = aInt.toLong()
    ulongs[4] = aInt.toULong()
    floats[4] = aInt.toFloat()
    doubles[4] = aInt.toDouble()

    bytes[5] = aUInt.toByte()
    ubytes[5] = aUInt.toUByte()
    shorts[5] = aUInt.toShort()
    ushorts[5] = aUInt.toUShort()
    ints[5] = aUInt.toInt()
    uints[5] = aUInt.toUInt()
    longs[5] = aUInt.toLong()
    ulongs[5] = aUInt.toULong()

    bytes[6] = aLong.toByte()
    ubytes[6] = aLong.toUByte()
    shorts[6] = aLong.toShort()
    ushorts[6] = aLong.toUShort()
    ints[6] = aLong.toInt()
    uints[6] = aLong.toUInt()
    longs[6] = aLong.toLong()
    ulongs[6] = aLong.toULong()
    floats[6] = aLong.toFloat()
    doubles[6] = aLong.toDouble()

    bytes[7] = aULong.toByte()
    ubytes[7] = aULong.toUByte()
    shorts[7] = aULong.toShort()
    ushorts[7] = aULong.toUShort()
    ints[7] = aULong.toInt()
    uints[7] = aULong.toUInt()
    longs[7] = aULong.toLong()
    ulongs[7] = aULong.toULong()

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