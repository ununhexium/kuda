package net.lab0.kuda

import jcuda.Pointer
import jcuda.driver.CUstream

data class KernelParameters(
    val gridDimX: Int = 1,
    val gridDimY: Int = 1,
    val gridDimZ: Int = 1,
    val blockDimX: Int = 1,
    val blockDimY: Int = 1,
    val blockDimZ: Int = 1,
    val sharedMemBytes: Int = 0,
    val hStream: CUstream? = null,
    val extra: Pointer? = null
)