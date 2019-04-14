package net.lab0.kuda

import java.nio.file.Files


object Config {
  val nvccPath
    get() = "/usr/local/cuda/bin/nvcc"
  val tmpFolder = Files.createTempDirectory(null)
}
