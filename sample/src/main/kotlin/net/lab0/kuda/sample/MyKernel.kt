package net.lab0.kuda.sample

import net.lab0.kuda.KernelParameters
import net.lab0.kuda.example.SaxpyKernelWrapper


fun main() {
  val myKernel = SaxpyKernelWrapper()
  val a = FloatArray(16) { it.toFloat() }
  val b = FloatArray(16) { 1.0f }
  val c = IntArray(100) { -1 }
  myKernel(KernelParameters(gridDimX = 1, blockDimX = 1), 10, 0.5f, a, b)
  println(c.joinToString())
}
