package net.lab0.kuda.sample.s1

import jcuda.Pointer
import jcuda.Sizeof
import jcuda.driver.CUdeviceptr
import jcuda.driver.CUfunction
import jcuda.driver.CUmodule
import jcuda.driver.JCudaDriver
import net.lab0.kuda.KudaContext
import net.lab0.kuda.copyToDevice
import net.lab0.kuda.generator.Global
import net.lab0.kuda.generator.Kernel
import net.lab0.kuda.generator.Result1
import net.lab0.kuda.generator.Return

/**
 * K1 with different return style
 */
@Kernel
class K3 : KudaContext() {
  @Global
  fun saxpy(n: Int, a: Float, x: FloatArray, y: FloatArray, @Return(size = 10) z: FloatArray) {
    val i: Int = blockIdx.x * blockDim.x + threadIdx.x
    if (i < n) z[i] = a * x[i] + y[i]
  }

  val ptxFileName = "some/resource"

  operator fun invoke(n: Int, a: Float, x: FloatArray, y: FloatArray): Result1<FloatArray> {

    // Load the ptx file.
    val module = CUmodule()
    JCudaDriver.cuModuleLoad(module, ptxFileName)

    // Obtain a function pointer to the function.
    val function = CUfunction()
    JCudaDriver.cuModuleGetFunction(function, module, "saxpy")

    // Allocate device memory
    val deviceX = x.copyToDevice()
    val deviceY = y.copyToDevice()

    val deviceZ = CUdeviceptr()
    JCudaDriver.cuMemAlloc(deviceZ, 10L * Sizeof.FLOAT)

    // Set up the kernel parameters: A pointer to an array
    // of pointers which point to the actual values.
    val kernelParameters = Pointer.to(
        Pointer.to(IntArray(1) { n }),
        Pointer.to(FloatArray(1) { a }),
        Pointer.to(deviceX),
        Pointer.to(deviceY),
        Pointer.to(deviceZ)
    )

    // Call the kernel function.
    val blockSizeX = 512
    val gridSizeX = Math.ceil(x.size.toDouble() / blockSizeX).toInt()
    JCudaDriver.cuLaunchKernel(
        function,
        gridSizeX, 1, 1, // Grid dimension
        blockSizeX, 1, 1, // Block dimension
        0, null // Kernel- and extra parameters
        , // Shared memory size and stream
        kernelParameters, null
    )
    JCudaDriver.cuCtxSynchronize()

    // Copy the device output to the host.
    val hostZ = FloatArray(10)
    JCudaDriver.cuMemcpyDtoH(
        Pointer.to(hostZ),
        deviceZ,
        (10 * Sizeof.FLOAT).toLong()
    )

    // Clean up.
    JCudaDriver.cuMemFree(deviceX)
    JCudaDriver.cuMemFree(deviceY)
    JCudaDriver.cuMemFree(deviceZ)

    return Result1(hostZ)
  }
}


fun main() {
  val kernel = K3()
  val xs = FloatArray(0)
  val ys = FloatArray(0)
  val result = kernel(10, 3.14f, xs, ys)
  println(result.a)
}