import net.lab0.kuda.Kotlin2Cuda
import net.lab0.kuda.annotation.Kernel
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
  compileOnly(project(":generator"))

  // TODO: have transitive dependencies
  implementation("com.google.guava:guava:27.1-jre")
  runtime("org.slf4j:slf4j-jdk14:1.8.0-beta4")

  implementation("org.jetbrains.kotlin:kotlin-stdlib-jdk8")
  compile("org.jetbrains.kotlin:kotlin-reflect")
  implementation("org.jcuda:jcuda:0.9.2") {
    isTransitive = false
  }

  implementation("net.lab0.kuda:annotation:0.1")
  implementation("net.lab0.kuda:s2s:0.1")

  // TEST

  testImplementation("org.assertj:assertj-core:3.12.2")

  val jUnitVersion = "5.3.1"
  testImplementation("org.junit.jupiter:junit-jupiter-api:$jUnitVersion")
  testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine:$jUnitVersion")
}


tasks {
  val kuda by registering(Task::class) {

    val javaSources = sourceSets.main.get().allJava
    logger.info(javaSources.srcDirs.joinToString { it.absolutePath })

    doLast {

      val outputPackage = "net.lab0.kuda.example"

      javaSources.srcDirs.flatMap { folder ->
        fileTree(folder).filter { file ->
          logger.info(file.absolutePath)
          file.readText().contains("import " + Kernel::class.qualifiedName)
        }.map { kernelFile ->
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
        }
      }
    }
  }

  withType<KotlinCompile> {
    dependsOn(kuda)
  }
}
