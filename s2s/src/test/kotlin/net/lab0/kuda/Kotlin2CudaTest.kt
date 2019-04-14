package net.lab0.kuda

import net.lab0.kuda.exception.CantConvert
import net.lab0.kuda.sample.s1.K1
import net.lab0.kuda.sample.s1.K2
import net.lab0.kuda.sample.s1.K3
import net.lab0.kuda.sample.s1.K4
import net.lab0.kuda.sample.s1.K5
import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows

internal class Kotlin2CudaTest {

  @Test
  fun `can convert a simple kotlin kernel to a C kernel`() {
    val cuda = loadAndTranspile(K1::class)

    val reference =
        """
          |extern "C"
          |
          |__global__
          |void saxpy(int n, float a, float * x, float * y)
          |{
          |  int i = blockIdx.x * blockDim.x + threadIdx.x;
          |  if (i < n) {
          |    y[i] = a * x[i] + y[i];
          |  };
          |}
        """.trimMargin()

    assertPtxEquals(cuda, reference)
  }

  @Test
  fun `can convert a kernel using 2D array`() {
    val cuda = loadAndTranspile(K2::class)

    val expected =
        """
          |extern "C"
          |
          |__global__
          |void MatAdd(float ** A, float ** B, float ** C)
          |{
          |  int i = threadIdx.x;
          |  int j = threadIdx.y;
          |  C[i][j] = A[i][j] + B[i][j];
          |}
        """.trimMargin()

    assertPtxEquals(cuda, expected)
  }

  @Test
  fun `output a meaningful error message when the @Global annotation is missing`() {
    val exception = assertThrows<CantConvert> {
      loadAndTranspile(K3::class)
    }

    assertThat(exception.message).contains(
        "There is no @Global function in the class K3."
    )
  }

  @Test
  fun `can add constant`() {
    val cuda = loadAndTranspile(K4::class)

    val expected =
        """
          |extern "C"
          |
          |__global__
          |void addN(int n, int * ns)
          |{
          |  int idx = blockIdx.x * blockDim.x + threadIdx.x;
          |  ns[idx] = ns[idx] + n;
          |}
        """.trimMargin()

    assertPtxEquals(cuda, expected)
  }

  @Test
  fun `output error message when the variable type is not specified`() {
    val exception = assertThrows<CantConvert> {
      loadAndTranspile(K5::class)
    }

    assertThat(exception.message).contains(
        "There is no type inference. As for now, you must specify the type of your left hand operand."
    )
  }
}
