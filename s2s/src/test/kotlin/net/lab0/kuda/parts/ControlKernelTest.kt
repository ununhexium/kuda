package net.lab0.kuda.parts

import net.lab0.kuda.StandardKernelTest
import net.lab0.kuda.sample.correct.ControlKernel

class ControlKernelTest: StandardKernelTest(ControlKernel::class) {

  override fun getEquivalentCKernel() =
      """
        |extern "C"
        |
        |__global__
        |void whileLoop(int * ints)
        |{
        |  int idx = blockDim.x;
        |  while (idx < 10) {
        |    ints[idx] = 1;
        |    if(5 == idx) {
        |      ints[idx] = -1;
        |    }
        |  }
        |}
      """.trimMargin()
}
