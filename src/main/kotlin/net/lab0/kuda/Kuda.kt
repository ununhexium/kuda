package net.lab0.kuda

import com.google.common.reflect.ClassPath
import net.lab0.kuda.annotation.Global
import net.lab0.kuda.annotation.Kernel
import java.lang.reflect.Method


class Kuda {
  fun scan(`package`: String): Iterable<Class<*>> {
    val classes = ClassPath
        .from(javaClass.classLoader)
        .getTopLevelClasses(`package`)
        .map { it.load() }
    return classes
        .filter { it.isAnnotationPresent(Kernel::class.java) }
  }

  fun findGlobalFunction(kernelClass: Class<*>): Method {
    return kernelClass.methods.first {
      it.isAnnotationPresent(Global::class.java)
    }
  }

  fun generateC(kernelClass: Class<*>): String {
    val global = findGlobalFunction(kernelClass)
    return """
      |__global__
      |void ${global.name}() {
      |
      |}
    """.trimMargin()
  }
}

