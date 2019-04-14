package net.lab0.kuda

import kastree.ast.Node
import net.lab0.kuda.exception.CantConvert

fun convertType(type: Node.Type): String {
  return when (type.ref) {
    is Node.TypeRef.Simple -> convertSimple(type.ref as Node.TypeRef.Simple)
    else -> throw CantConvert("Don't know how to convert $type")
  }
}

private val ARRAY_REGEX = Regex("(?<type>.+)Array")
private val UNSIGNED_REGEX = Regex("U(?<type>.+)")

fun convertSimple(simple: Node.TypeRef.Simple): String {
  val firstPiece = simple.pieces.first()

  return convertSimpleName(firstPiece.name, firstPiece) ?: throw CantConvert(simple)
}

fun convertSimpleName(name: String, firstPiece: Node.TypeRef.Simple.Piece): String? {
  val array = ARRAY_REGEX.matchEntire(name)
  if (array != null) {
    return convertSimpleName(array.groups["type"]!!.value, firstPiece) + " *"
  }

  val unsigned = UNSIGNED_REGEX.matchEntire(name)
  if (unsigned != null) {
    return "unsigned " + convertSimpleName(unsigned.groups["type"]!!.value, firstPiece)
  }

  return when (name) {
    "Boolean" -> "bool"
    "Byte" -> "char"
    "Short" -> "short"
    "Int" -> "int"
    "Long" -> "long"

    "Float" -> "float"
    "Double" -> "double"
    "Array" -> convertSimple(firstPiece.typeParams.first()?.ref as Node.TypeRef.Simple) + " *"
    else -> null
  }
}