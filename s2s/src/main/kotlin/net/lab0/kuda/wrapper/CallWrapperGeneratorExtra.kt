package net.lab0.kuda.wrapper

import com.squareup.kotlinpoet.ClassName
import com.squareup.kotlinpoet.CodeBlock
import com.squareup.kotlinpoet.ParameterizedTypeName
import com.squareup.kotlinpoet.asClassName
import jcuda.Sizeof
import kastree.ast.Node
import net.lab0.kuda.Config
import net.lab0.kuda.exception.CantConvert
import net.lab0.kuda.primitiveEquivalents
import kotlin.reflect.KClass
import com.squareup.kotlinpoet.ParameterizedTypeName.Companion.parameterizedBy
import com.squareup.kotlinpoet.TypeName
import com.squareup.kotlinpoet.asTypeName

fun Node.Type.toArrayType(): TypeName {
  val type = this.ref as Node.TypeRef.Simple
  return primitiveEquivalents[
      primitiveEquivalents.indexOfFirst {
        it.kClass.simpleName == type.pieces.first().name
      } + primitiveEquivalents.size / 2 // TODO: fix ugly hack
  ].kClass.asTypeName()
}

fun Node.Decl.Func.Param.isArray(): Boolean {
  // TODO: in what case is there more than 1 piece? Is that for classes generics?
  val ref = this.type!!.ref as Node.TypeRef.Simple
  val firstPiece = ref.pieces.first()
  // TODO: check string array ($ in the end)
  return firstPiece.name.matches(Regex("(U?(Byte|Int|Long|Short)|Boolean|Float|Double)Array|Array"))
}

fun Node.Decl.Func.Param.sizeOf() = findSizeOf(this)

fun Node.Type.asTypeName(): TypeName {
  val simple = this.ref as? Node.TypeRef.Simple
      ?: throw IllegalArgumentException("Can't extract classname from $this")

  // TODO: can there be more than 1 piece?
  if (simple.pieces.size > 1) throw CantConvert("Don't know how to handle multiple pieces $this")
  val firstPiece = simple.pieces.first()
  return firstPiece.asTypeName()
}

private fun Node.TypeRef.Simple.Piece.asTypeName(): TypeName {
  return primitiveEquivalents.firstOrNull {
    it.kClass.simpleName == this.name
  }?.kClass?.asClassName()
      ?: if (this.name == "Array") {
        return Array<Unit>::class.asClassName().parameterizedBy(
            *this.typeParams.map { it!!.asTypeName() }.toTypedArray()
        )
      } else throw CantConvert(this)
}


fun findSizeOf(param: Node.Decl.Func.Param): CodeBlock {
  return when (param.type!!.asTypeName()) {
    BooleanArray::class.asClassName() -> CodeBlock.of("%L", Config.booleanSizeInBytes)

    ByteArray::class.asClassName() -> CodeBlock.of("%T.BYTE", Sizeof::class)
    IntArray::class.asClassName() -> CodeBlock.of("%T.INT", Sizeof::class)
    LongArray::class.asClassName() -> CodeBlock.of("%T.LONG", Sizeof::class)
    ShortArray::class.asClassName() -> CodeBlock.of("%T.SHORT", Sizeof::class)

    DoubleArray::class.asClassName() -> CodeBlock.of("%T.DOUBLE", Sizeof::class)
    FloatArray::class.asClassName() -> CodeBlock.of("%T.FLOAT", Sizeof::class)
    else -> throw CantConvert("Can't get size of $param")
  }
}
