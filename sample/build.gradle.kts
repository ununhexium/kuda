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


sourceSets {
  main {
    java {
      srcDirs("${buildDir.absolutePath}/tmp/kapt/main/kotlinGenerated/")
    }
  }
}


dependencies {
  kapt(project(":generator"))
  compileOnly(project(":generator"))

  // TODO: have transitive dependencies
  implementation("com.google.guava:guava:27.1-jre")

  implementation("org.jetbrains.kotlin:kotlin-stdlib-jdk8")
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
      val kernels = javaSources.srcDirs.flatMap { folder ->
        fileTree(folder).filter { file ->
          file.readText().contains("import " + Kernel::class.qualifiedName)
        }
      }

      val cuda = kernels.map {
        K2C(it.readText()).transpile()
      }

      logger.info(
          "Generated cuda kernels:\n" +
              kernels.zip(cuda).joinToString("\n\n", prefix = "\n") {
                it.first.toString() + "\n" + it.second
              }
      )
    }
  }

  withType<KotlinCompile> {
    dependsOn(kuda)
  }
}
