package net.lab0.kuda

import kastree.ast.Node
import net.lab0.kuda.annotation.Return
import kotlin.reflect.KClass

fun Node.Decl.Structured.forHuman(): HumanReadableNode {
  return HumanReadableNode(this)
}

fun List<Node.Decl>.withAnnotation(name: String): List<Node.Decl.Structured> =
    this.mapNotNull {
      it as? Node.Decl.Structured
    }.filter {
      // TODO: filter by fully qualified name
      it.hasAnnotationNamed(name)
    }


fun Node.WithAnnotations.hasAnnotationNamed(name: String): Boolean {
  return anns.any { it.anns.any { it.names.any { it == name } } }
}


fun Node.WithAnnotations.hasAnnotation(kClass: KClass<*>): Boolean {
  // TODO: use canonical name
  return anns.any { it.anns.any { it.names.any { it == kClass.simpleName!! } } }
}


/**
 * Computes ceil(dividend / divisor)
 */
fun ceil(dividend: Int, divisor: Int) =
    if (dividend % divisor == 0) dividend / divisor
    else (dividend / divisor) + 1