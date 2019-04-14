package net.lab0.kuda

import com.github.difflib.DiffUtils
import com.github.difflib.UnifiedDiffUtils
import com.squareup.kotlinpoet.FunSpec
import com.squareup.kotlinpoet.ParameterSpec
import com.squareup.kotlinpoet.PropertySpec
import org.assertj.core.api.Assertions.assertThat
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import java.nio.file.Files
import java.nio.file.Paths
import kotlin.reflect.KClass

val log: Logger by lazy { LoggerFactory.getLogger("Helpers") }

fun loadSource(kClass: KClass<*>): String {
  return Paths
      .get(
          ".",
          "src",
          "test",
          "kotlin",
          *kClass.qualifiedName!!.split(".").dropLast(1).toTypedArray().also { it.last() },
          kClass.simpleName + ".kt"
      )
      .toFile()
      .readText()
}

fun loadAndTranspile(kClass: KClass<*>) =
    Kotlin2Cuda(loadSource(kClass)).transpile()

fun unifiedDiff(a: String, b: String): String {
  val diff = DiffUtils.diff(a, b)
  val unifiedDiff = UnifiedDiffUtils.generateUnifiedDiff("/dev/null", "/dev/null", a.split("\n"), diff, 2)
  return unifiedDiff.joinToString("\n") { it }
}

fun compileToPtx(source: String): String {
  val tmpDir = Files.createTempDirectory(null)
  val path = tmpDir.resolve("a.cu")
  path.toFile().writeText(source)

  val nvcc = Nvcc(path, outputFile = tmpDir.resolve("a.ptx"))
  val process = nvcc.executeProcess()
  process.waitFor()
  if (process.exitValue() != 0) {
    log.error(String(process.errorStream.readBytes()))
    log.info(String(process.inputStream.readBytes()))
  }
  return nvcc.outputFile.toFile().readText()
}

/**
 * Compiles and checks for output PTX file equality
 */
fun assertPtxEquals(a: String, b: String) {
  val ptxA = compileToPtx(a)
  val ptxB = compileToPtx(b)

  // unifiedDiff(a, b)
  assertThat(ptxA).isEqualTo("%s", ptxB)
}

fun List<FunSpec>.named(name: String) =
    this.first { it.name == name }

fun List<PropertySpec>.named(name: String) =
    this.first { it.name == name }

fun List<ParameterSpec>.named(name: String) =
    this.first { it.name == name }
