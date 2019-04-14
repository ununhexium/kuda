package net.lab0.kuda

import net.lab0.kuda.sample.s1.ArraysKernel
import org.junit.jupiter.api.Test

class ArraysKernelTest {
  @Test
  fun `can convert primitive array types`() {
    val cuda = loadAndTranspile(ArraysKernel::class)
    assertPtxEquals(
        cuda,
        """
          |extern "C"
          |
          |__global__ void arrays(
          |  bool * bools,
          |
          |  char * bytes,
          |  unsigned char * ubytes,
          |
          |  short * shorts,
          |  unsigned short * ushorts,
          |
          |  int * ints,
          |  unsigned int * uints,
          |
          |  long * longs,
          |  unsigned long * ulongs,
          |
          |  float * floats,
          |  double * doubles
          |) {
          |  bools[0] = true;
          |
          |  bytes[0] = -128;
          |  ubytes[0] = 255u;
          |
          |  shorts[0] = -32768;
          |  ushorts[0] = 65535u;
          |
          |  ints[0] = -2147483648;
          |  uints[0] = 4294967295u;
          |
          |  longs[0] = -9223372036854775807L -1L;
          |  ulongs[0] = 18446744073709551615u + 1u;
          |
          |  floats[0] = 1.175494351e-38f;
          |  doubles[0] = 1.7976931348623158e+308;
          |}
        """.trimMargin()
    )
  }
}