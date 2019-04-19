package net.lab0.kuda.parts

import net.lab0.kuda.StandardKernelTest
import net.lab0.kuda.sample.correct.PrimitivesCastKernel

internal class PrimitivesCastKernelTest : StandardKernelTest(PrimitivesCastKernel::class) {
  override fun getEquivalentCKernel() =
      """
        |
        |extern "C"
        |
        |__global__ void cast(
        |  char aByte,
        |  char * bytes,
        |
        |  short aShort,
        |  short * shorts,
        |
        |  int aInt,
        |  int * ints,
        |
        |  long aLong,
        |  long * longs,
        |
        |  float aFloat,
        |  float * floats,
        |
        |  double aDouble,
        |  double * doubles
        |) {
        |
        |  bytes[0] = (char) aByte;
        |  shorts[0] = (short) aByte;
        |  ints[0] = (int) aByte;
        |  longs[0] = (long) aByte;
        |  floats[0] = (float) aByte;
        |  doubles[0] = (double) aByte;
        |
        |  bytes[2] = (char) aShort;
        |  shorts[2] = (short) aShort;
        |  ints[2] = (int) aShort;
        |  longs[2] = (long) aShort;
        |  floats[2] = (float) aShort;
        |  doubles[2] = (double) aShort;
        |
        |  bytes[4] = (char) aInt;
        |  shorts[4] = (short) aInt;
        |  ints[4] = (int) aInt;
        |  longs[4] = (long) aInt;
        |  floats[4] = (float) aInt;
        |  doubles[4] = (double) aInt;
        |
        |  bytes[6] = (char) aLong;
        |  shorts[6] = (short) aLong;
        |  ints[6] = (int) aLong;
        |  longs[6] = (long) aLong;
        |  floats[6] = (float) aLong;
        |  doubles[6] = (double) aLong;
        |
        |  bytes[8] = (char) aFloat;
        |  shorts[8] = (short) aFloat;
        |  ints[8] = (int) aFloat;
        |  longs[8] = (long) aFloat;
        |  floats[8] = (float) aFloat;
        |  doubles[8] = (double) aFloat;
        |
        |  bytes[9] = (char) aDouble;
        |  shorts[9] = (short) aDouble;
        |  ints[9] = (int) aDouble;
        |  longs[9] = (long) aDouble;
        |  floats[9] = (float) aDouble;
        |  doubles[9] = (double) aDouble;
        |
        |}
        |
      """.trimMargin()
}