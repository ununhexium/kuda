import org.jetbrains.kotlin.gradle.tasks.KotlinCompile


plugins {
  kotlin("jvm")
  kotlin("kapt")
  id("maven-publish")
}


version = "0.1"


repositories {
  mavenCentral()
}


dependencies {
  implementation("org.jetbrains.kotlin:kotlin-stdlib-jdk8")
}


tasks {
  withType<KotlinCompile> {
    kotlinOptions.jvmTarget = "1.8"
  }

  val sourcesJar by registering(Jar::class) {
    archiveClassifier.set("sources")
    from(sourceSets.main.get().allSource)
  }
}


publishing {
  publications {
    create<MavenPublication>("annotation") {
      groupId = "net.lab0.kuda"
      artifactId = "annotation"
      version = "0.1"

      from(components["java"])
      val sourcesJar by tasks
      artifact(sourcesJar)
    }
  }
}
