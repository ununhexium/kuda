package net.lab0.kuda.exception

import net.lab0.kuda.HumanReadableNode

class CantConvert : IllegalStateException {
  constructor(any: Any) : super("Can't convert $any")
  constructor(message: String) : super(message)
  constructor(node: HumanReadableNode, message: String) : super("Can't convert $node. $message")
}
