package net.lab0.kuda

import net.lab0.kuda.sample.s1.DoNothing
import org.junit.jupiter.api.Test

class DoNothingTest {
  @Test
  fun `can do nothing`() {
    val cuda = loadAndTranspile(DoNothing::class)
    assertPtxEquals("""extern "C" __global__ void doNothing(void) {}""", cuda)
  }
}
