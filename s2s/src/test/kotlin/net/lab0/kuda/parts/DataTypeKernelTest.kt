package net.lab0.kuda.parts

import net.lab0.kuda.StandardKernelTest
import net.lab0.kuda.sample.correct.DataTypeKernel

class DataTypeKernelTest : StandardKernelTest(DataTypeKernel::class) {
  override fun getEquivalentCKernel() =
      """
        |extern "C"
        |
        |__global__ void dataTypes(
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
        |  bytes[0] = aByte;
        |
        |  shorts[0] = aShort;
        |
        |  ints[0] = aInt;
        |
        |  longs[0] = aLong;
        |
        |  floats[0] = aFloat;
        |  doubles[0] = aDouble;
        |}
      """.trimMargin()
}