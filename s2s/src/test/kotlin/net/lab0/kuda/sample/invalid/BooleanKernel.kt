package net.lab0.kuda.sample.invalid

import net.lab0.kuda.annotation.Global
import net.lab0.kuda.annotation.Kernel

/**
 * This is invalid as long as JCuda doesn't provide a way to transfer boolean arrays
 */
@Kernel
class BooleanKernel {
  @Global
  fun booleanStuff(data: BooleanArray) {
    val t: Boolean = true
    val f: Boolean = false

    data[0] = t
    data[1] = f
  }
}