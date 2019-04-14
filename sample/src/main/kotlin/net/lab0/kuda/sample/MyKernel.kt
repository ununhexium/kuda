package net.lab0.kuda.sample

import net.lab0.kuda.KernelParameters
import net.lab0.kuda.KudaContext
import net.lab0.kuda.annotation.Global
import net.lab0.kuda.annotation.Kernel
import net.lab0.kuda.annotation.Return
import net.lab0.kuda.example.MyKernelWrapper


@Kernel
class MyKernel : KudaContext() {
  @Global
  fun add(a: IntArray, b: IntArray, @Return c: IntArray) {
    val n: Int = blockIdx.x * blockDim.x + threadIdx.x
    c[n] = a[n] + b[n]
  }
}

fun main() {
  val myKernel = MyKernelWrapper()
  val a = (0..99).toList().toIntArray()
  val b = (0..99).toList().toIntArray()
  val c = IntArray(100) { -1 }
  myKernel(KernelParameters(gridDimX = 1, blockDimX = 1), a, b, c)
  println(c.joinToString())
}
