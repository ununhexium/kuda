package net.lab0.kuda.crippled

class FunSpec {

  fun builder(name: String): FunSpecBuilder {
    return FunSpecBuilder(name)
  }
}