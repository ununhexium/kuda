package net.lab0.kuda

import kastree.ast.Node
import kastree.ast.Visitor
import kastree.ast.psi.Parser
import net.lab0.kuda.exception.CantConvert
import java.lang.StringBuilder

class Kotlin2Cuda(private val source: String) {
  fun transpile(): String {
    val ast = Parser.parseFile(source)
    val output = StringBuilder()
    Visitor.visit(ast) { node, _ ->
      if (node is Node.File) {
        output.append(convertKotlinToC(node, 0))
      }
    }
    return output.toString()
  }

  private fun convertKotlinToC(node: Node.File, indent: Int): String {
    val structured = node.decls.first()
    return if (structured is Node.Decl.Structured && structured.hasAnnotationNamed("Kernel")) {
      convertClassToKernel(structured, indent)
    } else {
      throw CantConvert(node)
    }
  }

  private fun convertClassToKernel(node: Node.Decl.Structured, indent: Int): String {
    val global = node
        .members
        .mapNotNull { it as? Node.Decl.Func }
        .firstOrNull { it.hasAnnotationNamed("Global") }
        ?: throw CantConvert(node.forHuman(), "There is no @Global function in the class ${node.name}.")

    var out = """extern "C""""
    out += "\n\n__global__\n"

    out += "void ${global.name}("
    out +=
        global.params.joinToString(", ") { param ->
          convertType(param.type!!) + " " + param.name
        }
    out += ")"

    val block = global.body!!
    out += convertBodyToBlock(block, indent)

    return out
  }

  private fun getKernelClasses(): List<Node.WithAnnotations> {
    val ast = Parser.parseFile(source)
    return Visitor.filter(ast) { node ->
      node is Node.WithAnnotations && node.hasAnnotationNamed("Kernel")
    }.map {
      it as Node.WithAnnotations
    }
  }

  private fun convertBodyToBlock(body: Node.Decl.Func.Body, indent: Int): String {
    var out = "\n{\n"

    if (body is Node.Decl.Func.Body.Block) {
      out += convertBlock(body.block, indent + 2)
    }

    out += "\n}"

    return out
  }

  private fun convertBlock(block: Node.Block, indent: Int): String {
    return block.stmts.joinToString(";\n", postfix = ";") { stmt ->
      when (stmt) {
        is Node.Stmt.Decl -> convertDecl(stmt.decl, indent)
        is Node.Stmt.Expr -> convertExpr(stmt.expr, indent)
      }
    }
  }

  private fun convertExpr(expr: Node.Expr, indent: Int = 0): String {
    return when (expr) {
      is Node.Expr.BinaryOp -> convertBinaryOperation(expr, indent)
      is Node.Expr.Name -> convertName(expr, indent)
      is Node.Expr.ArrayAccess -> convertArrayAccess(expr, indent)
      is Node.Expr.If -> convertIf(expr, indent)
      is Node.Expr.Const -> expr.value
      is Node.Expr.While -> convertWhile(expr, indent)
      is Node.Expr.Brace -> convertBrace(expr, indent)
      is Node.Expr.UnaryOp -> convertUnaryOp(expr, indent)
      // TODO: more
      else -> throw CantConvert(expr)
    }.injectIndent(indent)
  }

  private fun convertUnaryOp(unaryOp: Node.Expr.UnaryOp, indent: Int): String {
    return when (unaryOp.oper.token) {
      Node.Expr.UnaryOp.Token.NEG -> " -" + convertExpr(unaryOp.expr)
      Node.Expr.UnaryOp.Token.INC -> convertExpr(unaryOp.expr) + "++"
      else -> throw CantConvert(unaryOp.oper.token)
    }
  }

  private fun convertBrace(expr: Node.Expr.Brace, indent: Int): String {
    return convertBlock(expr.block!!, indent)
  }

  private fun convertWhile(`while`: Node.Expr.While, indent: Int): String {
    val i = " ".repeat(indent)
    return """
      |while (${convertExpr(`while`.expr, 0)}) {
      |${convertExpr(`while`.body, indent)}
      |}
    """.trimMargin()
  }

  private fun convertIf(`if`: Node.Expr.If, indent: Int): String {
    val i = " ".repeat(indent)
    return """
      |if (${convertExpr(`if`.expr, 0)}) {
      |${convertExpr(`if`.body, indent)};
      |}
    """.trimMargin()
  }

  private fun convertArrayAccess(access: Node.Expr.ArrayAccess, indent: Int): String {
    return convertExpr(access.expr, indent) +
        "[" +
        access.indices.joinToString("\uD83D\uDCA9") {
          convertExpr(it)
        } +
        "]"
  }

  private fun convertName(name: Node.Expr.Name, indent: Int): String {
    return when (name) {
      name -> name.name
      else -> throw CantConvert(name)
    }
  }

  private fun convertBinaryOperation(binaryOp: Node.Expr.BinaryOp, indent: Int): String {
    return convertExpr(binaryOp.lhs) + convertOper(binaryOp.oper) + convertExpr(binaryOp.rhs)
  }

  private fun convertOper(oper: Node.Expr.BinaryOp.Oper, indent: Int = 0): String {
    return when (oper) {
      is Node.Expr.BinaryOp.Oper.Token -> convertToken(oper.token)
      else -> throw CantConvert(oper)
    }
  }

  private fun convertToken(token: Node.Expr.BinaryOp.Token): String {
    return when (token) {
      Node.Expr.BinaryOp.Token.ADD -> " + "
      Node.Expr.BinaryOp.Token.AND -> " && "
      Node.Expr.BinaryOp.Token.ASSN -> " = "
      Node.Expr.BinaryOp.Token.DOT -> "."
      Node.Expr.BinaryOp.Token.GTE -> " >= "
      Node.Expr.BinaryOp.Token.LT -> " < "
      Node.Expr.BinaryOp.Token.GT -> " > "
      Node.Expr.BinaryOp.Token.LTE -> " <= "
      Node.Expr.BinaryOp.Token.MUL -> " * "
      Node.Expr.BinaryOp.Token.SUB -> " - "
      // TODO: more
      else -> throw CantConvert(token)
    }
  }

  private fun convertDecl(decl: Node.Decl, indent: Int): String {
    return when (decl) {
      is Node.Decl.Property -> convertProperty(decl, indent)
      else -> throw CantConvert(decl)
    }.injectIndent(indent)
  }

  private fun convertProperty(property: Node.Decl.Property, indent: Int): String {
    val variable = property.vars.first()!!

    if (variable.type == null) {
      val origin = "Error on variable named ${variable.name}."
      throw CantConvert("$origin There is no type inference. As for now, you must specify the type of your left hand operand.")
    }

    var out = convertType(variable.type!!) + " " + variable.name

    val expr = property.expr
    if (expr != null) {
      out += " = " + convertExpr(expr)
    }

    return out
  }

  private fun Visitor.Companion.filter(node: Node, predicate: (Node) -> Boolean): List<Node> {
    val result = mutableListOf<Node>()
    Visitor.visit(node) { n, _ ->
      if (n != null && predicate(n)) {
        result.add(n)
      }
    }
    return result
  }

  private fun String.injectIndent(indent: Int): String {
    val i = " ".repeat(indent)
    return this.split("\n").joinToString("\n") { i + it }
  }

  private fun Node.WithAnnotations.hasAnnotationNamed(name: String): Boolean {
    return anns.any { it.anns.any { it.names.any { it == name } } }
  }

  private fun Node.WithModifiers.hasAnnotationNamed(name: String): Boolean {
    return anns.any { it.anns.any { it.names.any { it == name } } }
  }

}