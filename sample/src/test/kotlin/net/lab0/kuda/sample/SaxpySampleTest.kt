package net.lab0.kuda.sample

import net.lab0.kuda.KernelParameters
import net.lab0.kuda.example.SaxpySampleWrapper
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

internal class SaxpySampleTest {
  @Test
  fun `run saxpy`() {
    val saxpy = SaxpySampleWrapper()
    val a = 0.5f
    val x = FloatArray(116) { it.toFloat() }
    val y = FloatArray(116) { -1.0f }

    val res = saxpy(KernelParameters.for1D(x.size), 10, a, x, y)
    println(
        res.joinToString()
    )
  }
}