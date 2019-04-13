package net.lab0.kuda.sample

import net.lab0.kuda.KernelParameters
import net.lab0.kuda.KudaContext
import net.lab0.kuda.annotation.Global
import net.lab0.kuda.annotation.Kernel
import net.lab0.kuda.annotation.Return
import net.lab0.kuda.generated.kernel.MyKernelGenerated


@Kernel
class MyKernel : KudaContext() {
  @Global
  fun add(a: IntArray, b: IntArray, @Return c: IntArray) {
    val n: Int = blockIdx.x * blockDim.x + threadIdx.x
    c[n] = a[n] + b[n]
  }
}

fun main() {
  val myKernel = MyKernelGenerated()
  val a = (0..99).toList().toIntArray()
  val b = (0..99).toList().toIntArray()
  val c = IntArray(100) { -1 }
  myKernel(KernelParameters(gridDimX = a.size, blockDimX = 1), a, b, c)
  println(c.joinToString())
}

infix fun ClosedRange<Double>.step(step: Double): Iterable<Double> {
  require(start.isFinite())
  require(endInclusive.isFinite())
  require(step > 0.0) { "Step must be positive, was: $step." }
  val sequence = generateSequence(start) { previous ->
    if (previous == Double.POSITIVE_INFINITY) return@generateSequence null
    val next = previous + step
    if (next > endInclusive) null else next
  }
  return sequence.asIterable()
}
