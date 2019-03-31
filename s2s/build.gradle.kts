import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
  kotlin("jvm")
  kotlin("kapt")
}

repositories {
  mavenCentral()
}

val classifier = "linux-x86_64"

dependencies {

  implementation("com.github.cretz.kastree:kastree-ast-psi:0.4.0")
  testImplementation("com.google.guava:guava:27.1-jre")
  implementation("com.google.auto.service:auto-service:1.0-rc5")

  implementation("org.assertj:assertj-core:3.12.2")
  implementation("org.jetbrains.kotlin:kotlin-stdlib-jdk8")
  implementation("org.reflections:reflections:0.9.11")

  val jCudaVersion = "0.9.2"
  compile("org.jcuda:jcuda:0.9.2") {
    isTransitive = false
  }
  compile("org.jcuda", "jcuda-natives", jCudaVersion, classifier = classifier)

  implementation(project(":generator"))

  // TEST
  val jUnitVersion = "5.3.1"
  testImplementation("org.junit.jupiter:junit-jupiter-api:$jUnitVersion")
  testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine:$jUnitVersion")
}



tasks {
  withType<KotlinCompile>{
    kotlinOptions.jvmTarget = "1.8"
  }
}
