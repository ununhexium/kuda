package net.lab0.kuda.sample.s1

import net.lab0.kuda.KudaContext
import net.lab0.kuda.annotation.Global
import net.lab0.kuda.annotation.Kernel

@Kernel
class ArraysKernel: KudaContext() {
  @UseExperimental(ExperimentalUnsignedTypes::class)
  @Global
  fun arrays(
      bools: BooleanArray,

      bytes: ByteArray,
      ubytes: UByteArray,

      shorts: ShortArray,
      ushorts: UShortArray,

      ints: IntArray,
      uints: UIntArray,

      longs: LongArray,
      ulongs: ULongArray,

      floats: FloatArray,
      doubles: DoubleArray
  ) {
    bools[0] = true

    bytes[0] = -128
    ubytes[0] = 255u

    shorts[0] = -32768
    ushorts[0] = 65535u

    ints[0] = -2147483648
    uints[0] = 4294967295u

    longs[0] = -9223372036854775807L -1L // TODO: but why? To be able to check for underflow while keeping the compiler on the JVM?!?! Isn't that handeled as a BigInt internally in the compiler?
    ulongs[0] = 18446744073709551615u + 1u

    floats[0] = 1.175494351e-38f
    doubles[0] = 1.7976931348623158e+308
  }
}