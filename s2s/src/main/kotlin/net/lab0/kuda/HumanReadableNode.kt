package net.lab0.kuda

import kastree.ast.Node

class HumanReadableNode(private val node: Node) {
  override fun toString(): String {
    return when (node) {
      is Node.Decl.Structured -> node.name
      else -> node.toString()
    }
  }
}
