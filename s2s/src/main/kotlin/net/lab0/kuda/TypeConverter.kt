package net.lab0.kuda

import javassist.bytecode.stackmap.TypeData
import kastree.ast.Node
import net.lab0.kuda.exception.CantConvert
import kotlin.reflect.KClass

fun convertType(type: Node.Type): String {
  return when (type.ref) {
    is Node.TypeRef.Simple -> convertSimple(type.ref as Node.TypeRef.Simple)
    else -> throw CantConvert("Don't know how to convert $type")
  }
}

private val ARRAY_REGEX = Regex("(?<type>.+)Array")

data class PrimitiveEquivalent(val kClass: KClass<*>, val cName: String)

val primitiveEquivalents = listOf(
    PrimitiveEquivalent(Boolean::class, "bool"),

    PrimitiveEquivalent(Byte::class, "char"),
    PrimitiveEquivalent(Short::class, "short"),
    PrimitiveEquivalent(Int::class, "int"),
    PrimitiveEquivalent(Long::class, "long"),
    PrimitiveEquivalent(Float::class, "float"),
    PrimitiveEquivalent(Double::class, "double"),

    PrimitiveEquivalent(BooleanArray::class, "bool *"),

    PrimitiveEquivalent(ByteArray::class, "char *"),
    PrimitiveEquivalent(ShortArray::class, "short *"),
    PrimitiveEquivalent(IntArray::class, "int *"),
    PrimitiveEquivalent(LongArray::class, "long *"),
    PrimitiveEquivalent(FloatArray::class, "float *"),
    PrimitiveEquivalent(DoubleArray::class, "double *")
)

fun convertSimple(simple: Node.TypeRef.Simple): String {
  val firstPiece = simple.pieces.first()

  return convertSimpleName(firstPiece.name, firstPiece) ?: throw CantConvert(simple)
}

/**
 * Converts a Kotlin type name to the equivalent C type.
 */
fun convertSimpleName(name: String, firstPiece: Node.TypeRef.Simple.Piece): String {
  val array = ARRAY_REGEX.matchEntire(name)
  if (array != null) {
    return convertSimpleName(array.groups["type"]!!.value, firstPiece) + " *"
  }

  return when (name) {
    "Array" -> convertSimple(firstPiece.typeParams.first()?.ref as Node.TypeRef.Simple) + " *"
    else -> primitiveEquivalents.first {
      it.kClass.simpleName == name
    }.cName
  }
}

