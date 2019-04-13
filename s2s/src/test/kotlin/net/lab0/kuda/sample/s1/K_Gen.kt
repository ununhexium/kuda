package net.lab0.kuda.sample.s1

import com.google.common.io.Resources
import jcuda.Pointer
import jcuda.Sizeof
import jcuda.driver.CUdeviceptr
import jcuda.driver.CUfunction
import jcuda.driver.CUmodule
import jcuda.driver.CUstream
import jcuda.driver.JCudaDriver
import net.lab0.kuda.KernelParameters
import net.lab0.kuda.KudaContext
import net.lab0.kuda.copyToDevice
import net.lab0.kuda.annotation.Global
import net.lab0.kuda.annotation.Kernel
import net.lab0.kuda.annotation.Return

/**
 * K1 with different return style
 */
@Kernel
class K3Gen : KudaContext() {
  @Global
  fun saxpy(n: Int, a: Float, x: FloatArray, y: FloatArray, @Return z: FloatArray) {
    val i: Int = blockIdx.x * blockDim.x + threadIdx.x
    if (i < n) z[i] = a * x[i] + y[i]
  }

  val cudaResourceName = "some/resource.cuda"

  operator fun invoke(kernelParams: KernelParameters, n: Int, a: Float, x: FloatArray, y: FloatArray): FloatArray {

    val ptxFileName = compileCudaToPtx(Resources.getResource(cudaResourceName))

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
    JCudaDriver.cuLaunchKernel(
        function,
        kernelParams.gridDimX, kernelParams.gridDimY, kernelParams.gridDimZ, // Grid dimension
        kernelParams.blockDimX, kernelParams.blockDimY, kernelParams.blockDimZ, // Block dimension
        kernelParams.sharedMemBytes, kernelParams.hStream, // Shared memory size and stream
        kernelParameters, kernelParams.extra // Kernel- and extra parameters
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

    return hostZ
  }
}


fun main() {
  val kernel = K3Gen()
  val xs = FloatArray(0)
  val ys = FloatArray(0)
  val result = kernel(KernelParameters(gridDimX = 512), 10, 3.14f, xs, ys)
  println(result)
}
