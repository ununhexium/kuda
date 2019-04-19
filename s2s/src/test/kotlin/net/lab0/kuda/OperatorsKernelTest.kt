package net.lab0.kuda

import net.lab0.kuda.sample.OperatorsKernel
import org.junit.jupiter.api.Test

class OperatorsKernelTest {
  @Test
  fun `convert operators`() {
    val cuda = loadAndTranspile(OperatorsKernel::class)
    assertPtxEquals(
        cuda,
        """
          |extern "C"
          |
          |__global__ void operators(int i, long l, bool * b, long m) {
          |  long x0 = i + l;
          |  long x1 = i - l;
          |  long x2 = i * l;
          |  long x3 = i / l;
          |  long x4 = i % l;
          |
          |  long x5 = l;
          |
          |  long x6 = l & m;
          |  long x7 = l & m;
          |
          |  long x8 = l | m;
          |  long x9 = l | m;
          |
          |  long x10 = l ^ m;
          |  long x11 = l ^ m;
          |
          |  long sum = (
          |    x0 +
          |    x1 +
          |    x2 +
          |    x3 +
          |    x4 +
          |    x5++ +
          |    x5-- +
          |    ++x5 +
          |    --x5 +
          |    +x5 +
          |    -x5 +
          |    x6 +
          |    x7 +
          |    x8 +
          |    x9 +
          |    x10 +
          |    x11
          |  );
          |
          |  if(i < sum || i > sum) {
          |    b[0] = true;
          |  }
          |
          |  if (i >= sum && i <= sum) {
          |    b[0] = true;
          |  }
          |
          |  if (x5 == l || x5 != x0) {
          |    b[0] = true;
          |  }
          |
          |  if(!b[0]) {
          |    b[0] = true;
          |  }
          |}
        """.trimMargin()

    )
  }
}