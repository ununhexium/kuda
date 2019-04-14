package net.lab0.kuda.generator;

public class IllegalSourceCode extends Exception {
  public IllegalSourceCode(String message) {
    super(message);
  }

  public IllegalSourceCode(String message, Throwable cause) {
    super(message, cause);
  }

  public IllegalSourceCode(Throwable cause) {
    super(cause);
  }
}
