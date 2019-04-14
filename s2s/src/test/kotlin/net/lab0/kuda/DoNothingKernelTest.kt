package net.lab0.kuda

import net.lab0.kuda.sample.s1.DoNothingKernel
import org.junit.jupiter.api.Test

class DoNothingKernelTest {
  @Test
  fun `can do nothing`() {
    val cuda = loadAndTranspile(DoNothingKernel::class)
    assertPtxEquals("""extern "C" __global__ void doNothing(void) {}""", cuda)
  }
}
