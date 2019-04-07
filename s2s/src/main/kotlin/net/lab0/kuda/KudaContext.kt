package net.lab0.kuda

import com.google.common.io.Files
import com.google.common.io.Resources
import jcuda.Pointer
import jcuda.Sizeof
import jcuda.driver.CUcontext
import jcuda.driver.CUdevice
import jcuda.driver.CUdeviceptr
import jcuda.driver.JCudaDriver
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import java.io.IOException
import java.net.URL

open class KudaContext {
  companion object {
    private val log: Logger by lazy { LoggerFactory.getLogger(this::class.java.name) }
  }

  private var init = false
  protected fun init() {
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

  protected fun compileCudaToPtx(cudaResource: URL): String {
    val cudaContent = Resources.toString(cudaResource, Charsets.UTF_8)
    val tmpDir = Config.tmpFolder.resolve(cudaResource.path)
    val cudaSourceFile = tmpDir.resolve(cudaResource.file).toFile()
    cudaSourceFile.writeText(cudaContent)

    val cuFileName = cudaSourceFile.absolutePath.toString()
    val ptxFileName = cudaResource.file + ".ptx"
    val ptxFile = tmpDir.resolve(ptxFileName).toFile()
    if (ptxFile.exists()) {
      return ptxFileName
    }

    if (!cudaSourceFile.exists()) {
      throw IOException("Input file not found: $cuFileName")
    }
    val modelString = "-m" + System.getProperty("sun.arch.data.model")
    val command = arrayOf(
        Config.nvccPath,
        modelString,
        "-ptx",
        cudaSourceFile.path,
        "-o",
        ptxFileName
    )

    log.info("Executing\n${command.joinToString(" ")}")
    val process = Runtime.getRuntime().exec(command)

    val errorMessage = String(process.errorStream.readBytes())
    val outputMessage = String(process.inputStream.readBytes())

    val exitValue: Int
    try {
      exitValue = process.waitFor()
    } catch (e: InterruptedException) {
      Thread.currentThread().interrupt()
      throw IOException(
          "Interrupted while waiting for nvcc output", e
      )
    }


    if (exitValue != 0) {
      log.error("nvcc process exitValue $exitValue")
      log.error("Error  Message:\n$errorMessage")
      log.error("Output Message:\n$outputMessage")
      throw IOException(
          "Could not create .ptx file: $errorMessage"
      )
    }

    log.info("Finished creating PTX file")
    return ptxFileName
  }

  // TODO use long everywhere
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
