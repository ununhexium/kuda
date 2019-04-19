package net.lab0.kuda.sample

import net.lab0.kuda.KudaContext
import net.lab0.kuda.annotation.Global
import net.lab0.kuda.annotation.Kernel

@Kernel
class Matrix2DKernel: KudaContext() {
  @Global
  fun matrixAdd(A: Array<FloatArray>, B: Array<FloatArray>, C: Array<FloatArray>) {
    val i: Int = threadIdx.x
    val j: Int = threadIdx.y

    C[i][j] = A[i][j] + B[i][j]
  }
}


