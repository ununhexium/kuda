package net.lab0.kuda

import net.lab0.kuda.sample.DataTypeKernel
import org.junit.jupiter.api.Test

class DataTypeKernelTest {
  @Test
  fun `can convert primitive array types`() {
    val cuda = loadAndTranspile(DataTypeKernel::class)
    assertPtxEquals(
        cuda,
        """
          |extern "C"
          |
          |__global__ void dataTypes(
          |  bool aBool,
          |  bool * bools,
          |
          |  char aByte,
          |  char * bytes,
          |  unsigned char aUByte,
          |  unsigned char * ubytes,
          |
          |  short aShort,
          |  short * shorts,
          |  unsigned short aUShort,
          |  unsigned short * ushorts,
          |
          |  int aInt,
          |  int * ints,
          |  unsigned int aUInt,
          |  unsigned int * uints,
          |
          |  long aLong,
          |  long * longs,
          |  unsigned long aULong,
          |  unsigned long * ulongs,
          |
          |  float aFloat,
          |  float * floats,
          |
          |  double aDouble,
          |  double * doubles
          |) {
          |  bools[0] = aBool;
          |
          |  bytes[0] = aByte;
          |  ubytes[0] = aUByte;
          |
          |  shorts[0] = aShort;
          |  ushorts[0] = aUShort;
          |
          |  ints[0] = aInt;
          |  uints[0] = aUInt;
          |
          |  longs[0] = aLong;
          |  ulongs[0] = aULong;
          |
          |  floats[0] = aFloat;
          |  doubles[0] = aDouble;
          |}
        """.trimMargin()
    )
  }
}