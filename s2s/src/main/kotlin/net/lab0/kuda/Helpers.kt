package net.lab0.kuda

import kastree.ast.Node

fun Node.Decl.Structured.forHuman(): HumanReadableNode {
  return HumanReadableNode(this)
}
