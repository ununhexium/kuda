import net.lab0.kuda.Kotlin2Cuda
import net.lab0.kuda.annotation.Kernel
import net.lab0.kuda.exception.CantConvert
import net.lab0.kuda.wrapper.CallWrapperGenerator
import org.jetbrains.kotlin.gradle.tasks.KotlinCompile


buildscript {
  dependencies {
    classpath("org.reflections:reflections:0.9.11")
    classpath("net.lab0.kuda:s2s:0.1")
  }
  repositories {
    mavenCentral()
    mavenLocal()
  }
}


plugins {
  kotlin("jvm")
  id("maven-publish")
}


repositories {
  mavenLocal()
  mavenCentral()
}

val generatedResources = file("$buildDir/generated-resources")
val generatedSources = file("$buildDir/generated-sources")

sourceSets {
  main {
    java {
      srcDirs(generatedSources)
    }

    resources {
      srcDir(generatedResources)
    }
  }
}


dependencies {

  // TODO: have transitive dependencies
  implementation("com.google.guava:guava:27.1-jre")
  runtime("org.slf4j:slf4j-jdk14:1.8.0-beta4")

  implementation("org.jetbrains.kotlin:kotlin-stdlib-jdk8")
  compile("org.jetbrains.kotlin:kotlin-reflect")
  implementation("org.jcuda:jcuda:0.9.2") {
    isTransitive = false
  }

  implementation("net.lab0.kuda:s2s:0.1")

  // TEST

  testImplementation("org.assertj:assertj-core:3.12.2")

  val jUnitVersion = "5.3.1"
  testImplementation("org.junit.jupiter:junit-jupiter-api:$jUnitVersion")
  testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine:$jUnitVersion")
}


tasks {
  val headers by registering(Task::class) {
    val headerFiles = listOf(
        "/usr/local/cuda/include/crt/math_functions.h"
    )
    
    headerFiles.forEach {
      val f = file(it)
      f.readLines().filter {
        it.contains("__device_builtin__")
      }.map{
        it
            .replace("extern", "")
            .replace(Regex("_[a-zA-Z_]+( |;)"), "")
            .replace(Regex(" +"), " ")
      }.map{

      }
    }
  }

  val kuda by registering(Task::class) {

    val testSources = project(":s2s").sourceSets.test.get().allJava
    val javaSources = sourceSets.main.get().allJava
    val allSources = javaSources.srcDirs + testSources.srcDirs
    logger.info(allSources.joinToString { it.absolutePath })

    doLast {

      val outputPackage = "net.lab0.kuda.example"

      allSources.flatMap { folder ->
        fileTree(folder)
            .filter { file ->
              //              logger.info(file.absolutePath)
              file.readText().split("\n").any {
                it.matches(Regex("^import " + Kernel::class.qualifiedName))
              }
            }
            .filter { file ->
              logger.debug("considering $file")
              // only try to convert valid test kernels
              file.path.contains("correct")
            }
            .map { kernelFile ->
              logger.debug("Converting $kernelFile ...")
              try {
                val cuda = Kotlin2Cuda(kernelFile.readText()).transpile()
                val outputPackageFolder = outputPackage
                    .split(".")
                    .joinToString("/", postfix = "/")
                val cudaFile =
                    generatedResources.resolve(
                        outputPackageFolder + kernelFile.name + ".cu"
                    )
                if (!cudaFile.parentFile.exists()) {
                  cudaFile.parentFile.mkdirs()
                }
                cudaFile.writeText(cuda)
                logger.info("Generated ${kernelFile.name} cuda file to " + cudaFile.absolutePath)

                CallWrapperGenerator(kernelFile.readText(), outputPackage).callWrapperFile.writeTo(generatedSources)
                logger.info("Generated ${kernelFile.name} kernel wrapper file to " + generatedSources)

                logger.info(
                    "Generated cuda kernel:\n$kernelFile\n$cudaFile"
                )
              } catch (e: CantConvert) {
                logger.warn("Can't convert $kernelFile: ", e)
              }
            }
      }
    }
  }

  withType<KotlinCompile> {
    dependsOn(kuda)
  }
}
