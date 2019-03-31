package net.lab0.kuda.exception;

import kastree.ast.Node;
import org.jetbrains.annotations.NotNull;

public class CantConvert extends IllegalStateException {
  public CantConvert(@NotNull Object o) {
    super("Can't convert " + o);
  }

}
