package net.lab0.kuda.sample

import net.lab0.kuda.KernelParameters
import net.lab0.kuda.KudaContext
import net.lab0.kuda.annotation.Global
import net.lab0.kuda.annotation.Kernel
import net.lab0.kuda.generated.kernel.MyKernelGenerated

@Kernel
class MyKernel : KudaContext() {

  @Global
  fun theGlobal(n: Int, ns: IntArray) {
    val idx: Int = blockIdx.x * blockDim.x + threadIdx.x
//    ns[idx] = n
  }

}

fun main() {
  val myKernel = MyKernelGenerated()
  val ns = IntArray(1024) { 0 }
  myKernel(KernelParameters(gridDimX = 32, blockDimX = 32), 116, ns)
  println(ns.joinToString())
}
