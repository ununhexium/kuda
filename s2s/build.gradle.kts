import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
  kotlin("jvm")
  kotlin("kapt")
  id("java")
  id("maven-publish")
}

repositories {
  mavenLocal()
  mavenCentral()
}

val classifier = "linux-x86_64"

dependencies {

  implementation("com.github.cretz.kastree:kastree-ast-psi:0.4.0")

  implementation("net.lab0.kuda:annotation:0.1")

  implementation("org.jetbrains.kotlin:kotlin-stdlib-jdk8")
  implementation("org.reflections:reflections:0.9.11")
  implementation("org.slf4j:slf4j-api:1.8.0-beta4")


  val jCudaVersion = "0.9.2"
  implementation("org.jcuda:jcuda:0.9.2") {
    isTransitive = false
  }
  implementation("org.jcuda", "jcuda-natives", jCudaVersion, classifier = classifier)


  // TEST

  testImplementation("com.google.guava:guava:27.1-jre")
  testImplementation("org.assertj:assertj-core:3.12.2")
  testImplementation("org.slf4j:slf4j-jdk14:1.8.0-beta4")


  // TEST
  val jUnitVersion = "5.3.1"
  testImplementation("org.junit.jupiter:junit-jupiter-api:$jUnitVersion")
  testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine:$jUnitVersion")
}


publishing {
  publications {
    create<MavenPublication>("s2s") {
      groupId = "net.lab0.kuda"
      artifactId = "s2s"
      version = "0.1"

      from(components["java"])
    }
  }
}


tasks {
  withType<KotlinCompile>{
    kotlinOptions.jvmTarget = "1.8"
  }
}
