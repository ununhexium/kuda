package net.lab0.kuda

import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Disabled
import org.junit.jupiter.api.Test
import java.io.File

internal class K2CTest {
  @Test
  fun `can convert a simple kotlin kernel to a C kernel`() {
    // TODO: find how to get that file either via a class loader or by copying it to the resources with the annotation processor?
    val source = File(
        "/home/uuh/dev/116/kuda/s2s/src/test/kotlin/net/lab0/kuda/sample/s1/K1.kt"
    ).readText()

    val cuda = K2C.transpile(source)

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
    val source = File(
        "/home/uuh/dev/116/kuda/s2s/src/test/kotlin/net/lab0/kuda/sample/s1/K2.kt"
    ).readText()

    val cuda = K2C.transpile(source)

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
    val source = File(
        "/home/uuh/dev/116/kuda/s2s/src/test/kotlin/net/lab0/kuda/sample/s1/K3.kt"
    ).readText()

    val cuda = K2C.transpile(source)

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
}
