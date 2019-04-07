package net.lab0.kuda

import net.lab0.kuda.exception.CantConvert
import net.lab0.kuda.sample.s1.K0
import net.lab0.kuda.sample.s1.K1
import net.lab0.kuda.sample.s1.K2
import net.lab0.kuda.sample.s1.K3
import net.lab0.kuda.sample.s1.K4
import net.lab0.kuda.sample.s1.K5
import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Disabled
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import java.io.File
import java.nio.file.Paths
import kotlin.reflect.KClass

internal class K2CTest {
  companion object {
    fun loadSource(kClass: KClass<*>): String {
      return Paths
          .get(
              ".",
              "src",
              "test",
              "kotlin",
              *kClass.qualifiedName!!.split(".").dropLast(1).toTypedArray().also { it.last() },
              kClass.simpleName + ".kt"
          )
          .toFile()
          .readText()
    }
  }

  @Test
  fun `can convert a minimalist kernel`() {
    val source = loadSource(K0::class)

    val cuda = K2C(source).transpile()

    assertThat(cuda).isEqualTo(
        """
      |__global__
      |void count(int * ints)
      |{
      |  int idx = blockIdx.x * blockDim.x + threadIdx.x;
      |  ints[idx] = idx;
      |}
    """.trimMargin()
    )
  }

  @Test
  fun `can convert a simple kotlin kernel to a C kernel`() {
    // TODO: find how to get that file either via a class loader or by copying it to the resources with the annotation processor?
    val source = loadSource(K1::class)

    val cuda = K2C(source).transpile()

    assertThat(cuda).isEqualTo(
        """
          |__global__
          |void saxpy(int n, float a, float * x, float * y)
          |{
          |  int i = blockIdx.x * blockDim.x + threadIdx.x;
          |  if (i < n) {
          |    y[i] = a * x[i] + y[i];
          |  };
          |}
        """.trimMargin()
    )
  }

  @Test
  fun `can convert a kernel using 2D array`() {
    val source = loadSource(K2::class)

    val cuda = K2C(source).transpile()

    assertThat(cuda).isEqualTo(
        """
          |__global__
          |void MatAdd(float ** A, float ** B, float ** C)
          |{
          |  int i = threadIdx.x;
          |  int j = threadIdx.y;
          |  C[i][j] = A[i][j] + B[i][j];
          |}
        """.trimMargin()
    )
  }

  @Disabled
  @Test
  fun `can convert a kernel with loops`() {
    val source = loadSource(K3::class)

    val cuda = K2C(source).transpile()

    assertThat(
        cuda
    ).isEqualTo(
        """
          |__global__ void myKernel(cudaPitchedPtr devPitchedPtr, int width, int height, int depth)
          |{
          |    char* devPtr = devPitchedPtr.ptr;
          |    size_t pitch = devPitchedPtr.pitch;
          |    size_t slicePitch = pitch * height;
          |    for (int z = 0; z < depth; ++z) {
          |        char* slice = devPtr + z * slicePitch;
          |        for (int y = 0; y < height; ++y) {
          |            float* row = (float*)(slice + y * pitch);
          |            for (int x = 0; x < width; ++x) {
          |                float element = row[x];
          |            }
          |        }
          |    }
          |}
        """.trimMargin()
    )
  }

  @Test
  fun `output a meaningful error message when the @Global annotation is missing`() {
    val exception = assertThrows<CantConvert> {
      val source = loadSource(K3::class)
      K2C(source).transpile()
    }

    assertThat(exception.message).contains(
        "There is no @Global function in the class K3."
    )
  }

  @Test
  fun `can add constant`() {
    val source = loadSource(K4::class)

    val cuda = K2C(source).transpile()

    assertThat(cuda).isEqualTo(
        """
      |__global__
      |void addN(int n, int * ns)
      |{
      |  int idx = blockIdx.x * blockDim.x + threadIdx.x;
      |  ns[idx] = ns[idx] + n;
      |}
    """.trimMargin()
    )
  }

  @Test
  fun `output error message when the variable type is not specified`() {
    val exception = assertThrows<CantConvert> {
      val source = loadSource(K5::class)
      K2C(source).transpile()
    }

    assertThat(exception.message).contains(
        "There is no type inference. As for now, you must specify the type of your left hand operand."
    )
  }
}
