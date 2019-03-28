package test.sample.t2

import net.lab0.kuda.annotation.Global
import net.lab0.kuda.annotation.Kernel

/**
 * __global__
 * void foo() {
 * }
 */
@Kernel
class GlobalFoo {
  @Global
  fun foo() {

  }
}