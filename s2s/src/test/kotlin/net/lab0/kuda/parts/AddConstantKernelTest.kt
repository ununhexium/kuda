package net.lab0.kuda.parts

import net.lab0.kuda.StandardKernelTest
import net.lab0.kuda.sample.correct.AddConstantKernel

class AddConstantKernelTest : StandardKernelTest(AddConstantKernel::class) {
  override fun getEquivalentCKernel() =
      """
        |extern "C"
        |
        |__global__
        |void addN(int n, int * ns)
        |{
        |  int idx = blockIdx.x * blockDim.x + threadIdx.x;
        |  ns[idx] = ns[idx] + n;
        |}
      """.trimMargin()
}