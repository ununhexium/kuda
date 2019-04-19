package net.lab0.kuda.sample

import net.lab0.kuda.KernelParameters
import net.lab0.kuda.example.MandelbrotKernelWrapper
import java.awt.image.BufferedImage
import java.nio.file.Paths
import javax.imageio.ImageIO

fun main() {
  val mandelbrot = MandelbrotKernelWrapper()
  val side = 10000
  val step = 4.0 / side
  val area = side * side

  val position = { x: Int, y: Int -> x + y * side }

  val reals = DoubleArray(area)
  val imags = DoubleArray(area)
  (0 until side).forEach { x ->
    (0 until side).forEach { y ->
      val real = -2.0 + step * x
      val imag = -2.0 + step * y
      reals[position(x, y)] = real
      imags[position(x, y)] = imag
    }
  }
  var iterations = LongArray(reals.size)

  iterations = mandelbrot(KernelParameters.for1D(reals.size), reals, imags, 255, iterations)

  val max = iterations.max()!!

  val image = BufferedImage(side, side, BufferedImage.TYPE_INT_ARGB)
  val raster = image.raster
  val pixel = IntArray(4)
  (0 until side).forEach { x ->
    (0 until side).forEach { y ->
      val gray = iterations[position(x, y)].toInt()
      pixel[0] = gray
      pixel[1] = gray
      pixel[2] = gray
      pixel[3] = 255
      raster.setPixel(x, y, pixel)
    }
  }

  ImageIO.write(
      image,
      "png",
      Paths.get("/mnt", "ramdisk", "sample.png").toFile()
  )
}
