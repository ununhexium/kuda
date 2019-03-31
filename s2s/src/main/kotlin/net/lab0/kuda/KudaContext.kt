package net.lab0.kuda

import jcuda.Pointer
import jcuda.Sizeof
import jcuda.driver.CUcontext
import jcuda.driver.CUdevice
import jcuda.driver.CUdeviceptr
import jcuda.driver.JCudaDriver

open class KudaContext {
  private var init = false
  private fun init() {
    if (init) return

    JCudaDriver.setExceptionsEnabled(true)

    // Initialize the driver and create a context for the first device.
    JCudaDriver.cuInit(0)
    val device = CUdevice()
    JCudaDriver.cuDeviceGet(device, 0)
    val context = CUcontext()
    JCudaDriver.cuCtxCreate(context, 0, device)

    init = true
  }

  object blockIdx {
    const val x: Int = 0
    const val y: Int = 0
  }

  object blockDim {
    const val x: Int = 0
  }


  object threadIdx {
    const val x: Int = 0
    const val y: Int = 0
  }
}


fun FloatArray.copyToDevice(): CUdeviceptr {
  val devicePointer = CUdeviceptr()
  JCudaDriver.cuMemAlloc(devicePointer, this.cuSize())
  JCudaDriver.cuMemcpyHtoD(devicePointer, this.cuPtr(), this.cuSize())
  return devicePointer
}

fun FloatArray.cuSize() = Sizeof.FLOAT.toLong() * this.size

fun FloatArray.cuPtr() = Pointer.to(this)
