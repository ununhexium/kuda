package net.lab0.kuda

import kastree.ast.Node
import kastree.ast.Visitor
import kastree.ast.psi.Parser
import org.assertj.core.api.Assertions
import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import java.lang.StringBuilder

internal class TypeConverterKtTest {
  companion object {
    private fun getMatrix(sample: String): Node.TypeRef.Simple {
      lateinit var matrix: Node.TypeRef.Simple
      val ast = Parser.parseFile(sample)
      Visitor.visit(ast) { node, _ ->
        if (node is Node.File) {
          val p = node.decls.first() as Node.Decl.Property
          matrix = p.vars.first()!!.type!!.ref as Node.TypeRef.Simple
        }
      }
      return matrix
    }
  }

  @Test
  fun `can convert a single type`() {
    assertThat(
        convertSimple(
            Node.TypeRef.Simple(listOf(Node.TypeRef.Simple.Piece("Boolean", listOf())))
        )
    ).isEqualTo(
        "bool"
    )
  }

  @Test
  fun `can convert an array type`() {
    assertThat(
        convertSimple(
            Node.TypeRef.Simple(listOf(Node.TypeRef.Simple.Piece("BooleanArray", listOf())))
        )
    ).isEqualTo(
        "bool *"
    )
  }

  @Test
  fun `can convert a generic array type`() {
    assertThat(
        convertSimple(
            getMatrix("lateinit val b: Array<Boolean>")
        )
    ).isEqualTo(
        "bool *"
    )
  }

  @Test
  fun `can convert a matrix`() {
    assertThat(
        convertSimple(
            getMatrix("lateinit var b: Array<BooleanArray>")
        )
    ).isEqualTo(
        "bool * *"
    )
  }

  @Test
  fun `can convert a 3D matrix`() {
    val sample = """
      |lateinit var b: Array<Array<BooleanArray>>
    """.trimMargin()

    val matrix: Node.TypeRef.Simple = getMatrix(sample)

    assertThat(
        convertSimple(matrix)
    ).isEqualTo(
        "bool * * *"
    )
  }
}