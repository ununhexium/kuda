package net.lab0.kuda.parts

import net.lab0.kuda.StandardKernelTest
import net.lab0.kuda.sample.invalid.BooleanKernel

class BooleanKernelTest: StandardKernelTest(BooleanKernel::class) {
  override fun getEquivalentCKernel() =
      """
        |extern "C"
        |
        |__global__
        |void booleanStuff(bool * data)
        |{
        |  bool t = true;
        |  bool f = false;
        |  data[0] = t;
        |  data[1] = f;
        |}
      """.trimMargin()
}
