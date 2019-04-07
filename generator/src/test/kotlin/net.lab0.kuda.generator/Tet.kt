//package net.lab0.kuda.generator
//
//import com.google.common.truth.Truth.assertThat
//import com.google.testing.compile.Compiler
//import com.google.testing.compile.JavaFileObjects
//import org.junit.jupiter.api.Test
//
//
//class Tet {
//  @Test
//  fun `foo`() {
//    val helloWorld = JavaFileObjects.forResource("HelloWorld.java")
//    val compilation = Compiler.javac()
//        .withProcessors(KernelGenerator())
//        .compile(helloWorld)
//
//    assertThat(compilation).failed()
//    assertThat(compilation)
//        .hadErrorContaining("No types named HelloWorld!")
//        .inFile(helloWorld)
//        .onLine(23)
//        .atColumn(5);
//  }
//}