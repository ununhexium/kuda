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
    "Boolean" -> "bool"
    "Double" -> "double"
    "Float" -> "float"
    "Int" -> "int"
    "Long" -> "long"
    "BooleanArray" -> "bool *"
    "ByteArray" -> "char *"
    "UByteArray" -> "unsigned char *"
    "DoubleArray" -> "double *"
    "FloatArray" -> "float *"
    "IntArray" -> "int *"
    "UIntArray" -> "unsigned int *"
    "LongArray" -> "long *"
    "ULongArray" -> "unsigned long *"
    "ShortArray" -> "short *"
    "UShortArray" -> "unsigned short *"
    "Array" -> convertSimple(firstPiece.typeParams.first()?.ref as Node.TypeRef.Simple) + "*"
    else -> throw IllegalStateException("Don't know how to convert $simple")
  }
}
