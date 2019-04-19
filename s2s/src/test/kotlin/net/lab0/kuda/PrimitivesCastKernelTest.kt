package net.lab0.kuda

import net.lab0.kuda.sample.PrimitivesCastKernel
import org.junit.jupiter.api.Test

internal class PrimitivesCastKernelTest {
  @Test
  fun `cast primitive types`() {
    assertPtxEquals(
        loadAndTranspile(PrimitivesCastKernel::class),
        """
          |
          |extern "C"
          |
          |__global__ void cast(
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
          |
          |  bytes[0] = (char) aByte;
          |  ubytes[0] = (unsigned char) aByte;
          |  shorts[0] = (short) aByte;
          |  ushorts[0] = (unsigned short) aByte;
          |  ints[0] = (int) aByte;
          |  uints[0] = (unsigned int) aByte;
          |  longs[0] = (long) aByte;
          |  ulongs[0] = (unsigned long) aByte;
          |  floats[0] = (float) aByte;
          |  doubles[0] = (double) aByte;
          |
          |  bytes[1] = (char) aUByte;
          |  ubytes[1] = (unsigned char) aUByte;
          |  shorts[1] = (short) aUByte;
          |  ushorts[1] = (unsigned short) aUByte;
          |  ints[1] = (int) aUByte;
          |  uints[1] = (unsigned int) aUByte;
          |  longs[1] = (long) aUByte;
          |  ulongs[1] = (unsigned long) aUByte;
          |
          |  bytes[2] = (char) aShort;
          |  ubytes[2] = (unsigned char) aShort;
          |  shorts[2] = (short) aShort;
          |  ushorts[2] = (unsigned short) aShort;
          |  ints[2] = (int) aShort;
          |  uints[2] = (unsigned int) aShort;
          |  longs[2] = (long) aShort;
          |  ulongs[2] = (unsigned long) aShort;
          |  floats[2] = (float) aShort;
          |  doubles[2] = (double) aShort;
          |
          |  bytes[3] = (char) aUShort;
          |  ubytes[3] = (unsigned char) aUShort;
          |  shorts[3] = (short) aUShort;
          |  ushorts[3] = (unsigned short) aUShort;
          |  ints[3] = (int) aUShort;
          |  uints[3] = (unsigned int) aUShort;
          |  longs[3] = (long) aUShort;
          |  ulongs[3] = (unsigned long) aUShort;
          |
          |  bytes[4] = (char) aInt;
          |  ubytes[4] = (unsigned char) aInt;
          |  shorts[4] = (short) aInt;
          |  ushorts[4] = (unsigned short) aInt;
          |  ints[4] = (int) aInt;
          |  uints[4] = (unsigned int) aInt;
          |  longs[4] = (long) aInt;
          |  ulongs[4] = (unsigned long) aInt;
          |  floats[4] = (float) aInt;
          |  doubles[4] = (double) aInt;
          |
          |  bytes[5] = (char) aUInt;
          |  ubytes[5] = (unsigned char) aUInt;
          |  shorts[5] = (short) aUInt;
          |  ushorts[5] = (unsigned short) aUInt;
          |  ints[5] = (int) aUInt;
          |  uints[5] = (unsigned int) aUInt;
          |  longs[5] = (long) aUInt;
          |  ulongs[5] = (unsigned long) aUInt;
          |
          |  bytes[6] = (char) aLong;
          |  ubytes[6] = (unsigned char) aLong;
          |  shorts[6] = (short) aLong;
          |  ushorts[6] = (unsigned short) aLong;
          |  ints[6] = (int) aLong;
          |  uints[6] = (unsigned int) aLong;
          |  longs[6] = (long) aLong;
          |  ulongs[6] = (unsigned long) aLong;
          |  floats[6] = (float) aLong;
          |  doubles[6] = (double) aLong;
          |
          |  bytes[7] = (char) aULong;
          |  ubytes[7] = (unsigned char) aULong;
          |  shorts[7] = (short) aULong;
          |  ushorts[7] = (unsigned short) aULong;
          |  ints[7] = (int) aULong;
          |  uints[7] = (unsigned int) aULong;
          |  longs[7] = (long) aULong;
          |  ulongs[7] = (unsigned long) aULong;
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
          |
        """.trimMargin()
    )
  }
}