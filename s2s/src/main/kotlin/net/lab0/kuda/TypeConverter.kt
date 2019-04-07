package net.lab0.kuda

import kastree.ast.Node
import java.lang.IllegalStateException

fun convertType(type: Node.Type): String {
  return when (type.ref) {
    is Node.TypeRef.Simple -> convertSimple(type.ref as Node.TypeRef.Simple)
    else -> throw IllegalStateException("Don't know how to convert $type")
  }
}

fun convertSimple(simple: Node.TypeRef.Simple): String {
  val firstPiece = simple.pieces.first()
  return when (firstPiece.name) {
    "Int" -> "int"
    "Float" -> "float"
    "FloatArray" -> "float *"
    "IntArray" -> "int *"
    "Array" -> convertSimple(firstPiece.typeParams.first()?.ref as Node.TypeRef.Simple) + "*"
    else -> throw IllegalStateException("Don't know how to convert $simple")
  }
}
