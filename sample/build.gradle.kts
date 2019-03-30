
plugins {
  kotlin("jvm")
  kotlin("kapt")
}


kapt {
}

repositories {
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
  implementation("org.jetbrains.kotlin:kotlin-stdlib-jdk8")
}
