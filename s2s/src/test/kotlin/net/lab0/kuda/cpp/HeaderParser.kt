package net.lab0.kuda.cpp

import org.antlr.v4.runtime.CharStreams
import org.antlr.v4.runtime.CommonTokenStream
import org.antlr.v4.runtime.ParserRuleContext
import org.antlr.v4.runtime.RuleContext
import org.antlr.v4.runtime.tree.ParseTree
import org.junit.jupiter.api.Disabled
import org.junit.jupiter.api.Test
import java.io.File
import java.nio.charset.Charset


private val CPP14Parser.DeclspecifierseqContext.lastDeclspecifierseq: String
  get() =
    if (this.declspecifierseq() == null) {
      this.declspecifier().readableText
    } else {
      this.declspecifierseq().lastDeclspecifierseq
    }

private val ParseTree.readableText: String
  get() =
    if (childCount == 0) {
      text
    } else {
      (0 until childCount).joinToString(" ") {
        getChild(it)?.readableText ?: ""
      }
    }

class HeaderParser {

  class Visitor : CPP14BaseVisitor<Unit>() {
    override fun visitSimpledeclaration(ctx: CPP14Parser.SimpledeclarationContext) {
      println(ctx.readableText)
      println(ctx.declspecifierseq().lastDeclspecifierseq)
    }
  }

  @Disabled
  @Test
  fun `parser header`() {
    val lexer = CPP14Lexer(CharStreams.fromFileName("/usr/local/cuda/include/crt/math_functions.h"))
    val tokens = CommonTokenStream(lexer)
    val parser = CPP14Parser(tokens)
    Visitor().visit(parser.translationunit())
  }
}