import net.lab0.kuda.K2C
import net.lab0.kuda.annotation.Kernel
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
  kotlin("kapt")
  id("maven-publish")
}


kapt {
}


repositories {
  mavenLocal()
  mavenCentral()
}

val generatedResources = file("$buildDir/generated-resources")

sourceSets {
  main {
    java {
      srcDirs("${buildDir.absolutePath}/tmp/kapt/main/kotlinGenerated/")
    }

    resources {
      srcDir(generatedResources)
    }
  }
}


dependencies {
  kapt(project(":generator"))
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
}


tasks {
  val kuda by registering(Task::class) {
    doLast {
      val javaSources = sourceSets.main.get().allJava

      javaSources.srcDirs.flatMap { folder ->
        fileTree(folder).filter { file ->
          file.readText().contains("import " + Kernel::class.qualifiedName)
        }.map { kernel ->
          val cuda = K2C(kernel.readText()).transpile()
          val cudaFile = generatedResources.resolve("net/lab0/kuda/kernel/" + kernel.name + ".cu")
          if (!cudaFile.parentFile.exists()) {
            cudaFile.parentFile.mkdirs()
          }
          cudaFile.writeText(cuda)

          logger.info(
              "Generated cuda kernel:\n$kernel\n$cudaFile"
          )
        }
      }

    }
  }

  withType<KotlinCompile> {
    dependsOn(kuda)
  }
}
