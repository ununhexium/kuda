package net.lab0.kuda

import com.github.difflib.DiffUtils
import com.github.difflib.UnifiedDiffUtils
import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

internal class TestHelpersKtTest {
  @Test
  fun `unified diff`() {
    val a =
        """
        extern "C"
        __global__
        void whileLoop()
        {
          bool b;
          while (false) {
            b = true;
          };
        }
        """.trimIndent()

    val b =
        """
        extern "C"
        __global__ void whileLoop()
        {
          bool b;

          while (false) {
            b = true;
          }
        }
        """.trimIndent()

    val reference =
        """
        |--- /dev/null
        |+++ /dev/null
        |@@ -1,9 +1,9 @@
        | extern "C"
        |-__global__
        |-void whileLoop()
        |+__global__ void whileLoop()
        | {
        |   bool b;
        |+
        |   while (false) {
        |     b = true;
        |-  };
        |+  }
        | }
        """.trimMargin()


    //generating unified diff format

    assertThat(unifiedDiff(a,b)).isEqualTo("%S", reference)
  }
}