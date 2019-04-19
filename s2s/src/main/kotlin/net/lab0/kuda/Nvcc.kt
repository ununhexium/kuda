package net.lab0.kuda

import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths

data class Nvcc(
    val ptx: Path,
    //Files.createTempDirectory(null).resolve(ptx.fileName.toFile().name)
    val outputFile: Path = Paths.get("/tmp", "ramdisk"),
    val machine: String = System.getProperty("sun.arch.data.model") ?: "64",
    val nvccBinary: String = Config.nvccPath
) {
  fun executeProcess(): Process {
    return Runtime.getRuntime().exec(
        arrayOf(
            nvccBinary,
            "--machine", machine,
            "--ptx", ptx.toAbsolutePath().toString(),
            "--output-file", outputFile.toAbsolutePath().toString()
        )
    )
  }
}
