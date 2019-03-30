import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
  kotlin("jvm")
  kotlin("kapt")
}

repositories {
  mavenCentral()
}

dependencies {
  implementation("org.jetbrains.kotlin:kotlin-stdlib-jdk8")
  compile("com.squareup:kotlinpoet:1.2.0")
  compile("com.google.auto.service:auto-service:1.0-rc5")
  kapt("com.google.auto.service:auto-service:1.0-rc5")
}

tasks {
  withType<KotlinCompile>{
    kotlinOptions.jvmTarget = "1.8"
  }
}
