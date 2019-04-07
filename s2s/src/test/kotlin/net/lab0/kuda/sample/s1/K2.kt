package net.lab0.kuda.sample.s1

import net.lab0.kuda.KudaContext
import net.lab0.kuda.annotation.Global
import net.lab0.kuda.annotation.Kernel

@Kernel
class K2: KudaContext() {
  @Global
  fun MatAdd(A: Array<FloatArray>, B: Array<FloatArray>, C: Array<FloatArray>) {
    val i: Int = threadIdx.x
    val j: Int = threadIdx.y

    C[i][j] = A[i][j] + B[i][j]
  }
}


