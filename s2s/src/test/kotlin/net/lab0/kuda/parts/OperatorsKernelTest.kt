package net.lab0.kuda.parts

import net.lab0.kuda.StandardKernelTest
import net.lab0.kuda.sample.correct.OperatorsKernel

class OperatorsKernelTest: StandardKernelTest(OperatorsKernel::class) {
  override fun getEquivalentCKernel() =
      """
        |extern "C"
        |
        |__global__ void operators(int i, long l, short * holder, long m) {
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
        |    holder[0] = 1;
        |  }
        |
        |  if (i >= sum && i <= sum) {
        |    holder[0] = 1;
        |  }
        |
        |  if (x5 == l || x5 != x0) {
        |    holder[0] = 1;
        |  }
        |
        |  if(!false) {
        |    holder[0] = -1;
        |  }
        |}
      """.trimMargin()
}
