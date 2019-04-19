//package net.lab0.kuda.parts
//
//import net.lab0.kuda.StandardKernelTest
//import net.lab0.kuda.sample.correct.Matrix2DKernel
//
//internal class Matrix2DKernelTest: StandardKernelTest(Matrix2DKernel::class) {
//  override fun getEquivalentCKernel() =
//      """
//        |extern "C"
//        |
//        |__global__
//        |void matrixAdd(float * * A, float * * B, float * * C)
//        |{
//        |  int i = threadIdx.x;
//        |  int j = threadIdx.y;
//        |  C[i][j] = A[i][j] + B[i][j];
//        |}
//      """.trimMargin()
//}