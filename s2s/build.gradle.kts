import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
  kotlin("jvm")
  kotlin("kapt")
  id("java")
  id("maven-publish")
  id("antlr")
}

repositories {
  mavenLocal()
  mavenCentral()
}

val generatedSources = file("$buildDir/generated-sources")


sourceSets {
  main {
    java {
      srcDirs(generatedSources)
    }
  }
}


dependencies {
  implementation("com.github.cretz.kastree:kastree-ast-psi:0.4.0")

  implementation("com.squareup:kotlinpoet:1.2.0")

  antlr("org.antlr:antlr4:4.7.2")

  implementation("org.jetbrains.kotlin:kotlin-stdlib-jdk8")
  implementation("org.jetbrains.kotlin:kotlin-reflect")

  implementation("org.reflections:reflections:0.9.11")

  implementation("org.slf4j:slf4j-api:1.8.0-beta4")


  val jCudaVersion = "0.9.2"
  val classifier = "linux-x86_64"
  implementation("org.jcuda", "jcuda", jCudaVersion) {
    isTransitive = false
  }
  implementation("org.jcuda", "jcuda-natives", jCudaVersion, classifier = classifier)


  // TEST

  testImplementation("com.github.wumpz:diffutils:2.2")
  testImplementation("com.google.guava:guava:27.1-jre")
  testImplementation("org.assertj:assertj-core:3.12.2")
  testImplementation("org.slf4j:slf4j-jdk14:1.8.0-beta4")

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
  withType<KotlinCompile> {
    kotlinOptions.jvmTarget = "1.8"
  }

  withType<AntlrTask> {
    arguments = arguments + listOf("-visitor", "-long-messages")
  }
}



