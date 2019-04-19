package net.lab0.kuda.parts

import net.lab0.kuda.assertPtxEquals
import net.lab0.kuda.loadAndTranspile
import net.lab0.kuda.sample.correct.DoNothingKernel
import org.junit.jupiter.api.Test

class DoNothingKernelTest {
  @Test
  fun `can do nothing`() {
    val cuda = loadAndTranspile(DoNothingKernel::class)
    assertPtxEquals("""extern "C" __global__ void doNothing(void) {}""", cuda)
  }
}
