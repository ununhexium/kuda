import org.jetbrains.kotlin.gradle.tasks.KotlinCompile


plugins {
  kotlin("jvm")
  kotlin("kapt")
}


version = "0.1"


repositories {
  mavenLocal()
  mavenCentral()
}


dependencies {
  implementation("com.google.auto.service:auto-service:1.0-rc5")
  implementation("com.squareup:kotlinpoet:1.2.0")

  implementation("net.lab0.kuda:annotation:0.1")
  implementation("net.lab0.kuda:s2s:0.1")

  implementation("org.jcuda:jcuda:0.9.2") {
    isTransitive = false
  }

  implementation("org.jetbrains.kotlin:kotlin-stdlib-jdk8")

  implementation("org.slf4j:slf4j-api:1.8.0-beta4")

  kapt("com.google.auto.service:auto-service:1.0-rc5")

  // TEST

  testImplementation("com.google.testing.compile:compile-testing:0.15")
  val jUnitVersion = "5.3.1"
  testImplementation("org.junit.jupiter:junit-jupiter-api:$jUnitVersion")
  testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine:$jUnitVersion")
}


tasks {
  withType<KotlinCompile> {
    kotlinOptions.jvmTarget = "1.8"
  }

  withType<Test> {
    useJUnitPlatform()
  }
}
