package net.lab0.kuda.sample

import net.lab0.kuda.annotation.Global
import net.lab0.kuda.annotation.Kernel

@Suppress("ConvertTwoComparisonsToRangeCheck")
@Kernel
class OperatorsKernel {
  @Global
  fun operators(i: Int, l: Long, b: BooleanArray, m: Long) {
    val x0: Long = i + l
    val x1: Long = i - l
    val x2: Long = i * l
    val x3: Long = i / l
    val x4: Long = i % l
    var x5: Long = l

    val x6: Long = l.and(m)
    val x7: Long = l and m

    val x8: Long = l.or(m)
    val x9: Long = l or m

    val x10: Long = l.xor(m)
    val x11: Long = l xor m

    // using the vars to be sure they won't be optimized away
    val sum: Long = (
        x0 +
            x1 +
            x2 +
            x3 +
            x4 +
            x5++ +
            x5-- +
            ++x5 +
            --x5 +
            +x5 +
            -x5 +
            x6 +
            x7 +
            x8 +
            x9 +
            x10 +
            x11
        )

    if (i < sum || i > sum) {
      b[0] = true
    }

    if (i >= sum && i <= sum) {
      b[0] = true
    }

    if (x5 == l || x5 != x0) {
      b[0] = true
    }

    if (!b[0]) {
      b[0] = true
    }

  }
}