package net.lab0.kuda.generator

import javax.lang.model.element.Element
import javax.lang.model.element.ElementVisitor
import javax.lang.model.element.ExecutableElement
import javax.lang.model.element.PackageElement
import javax.lang.model.element.TypeElement
import javax.lang.model.element.TypeParameterElement
import javax.lang.model.element.VariableElement
import javax.lang.model.util.AbstractElementVisitor8

class MyVisitor : AbstractElementVisitor8<Void?, Void?>() {
  val executables = mutableListOf<ExecutableElement>()
  val types = mutableListOf<TypeElement>()
  val variables = mutableListOf<VariableElement>()
  val typeParameters = mutableListOf<TypeParameterElement>()
  override fun visitType(e: TypeElement, p: Void?): Void? {
    types.add(e)
    return null
  }

  override fun visitTypeParameter(e: TypeParameterElement, p: Void?): Void? {
    typeParameters.add(e)
    return null
  }

  override fun visitExecutable(e: ExecutableElement, p: Void?): Void? {
    executables.add(e)
    return null
  }

  override fun visitVariable(e: VariableElement, p: Void?): Void? {
    variables.add(e)
    return null
  }

  override fun visitUnknown(e: Element?, p: Void?): Void? {
//        debug.append(e.toString()).append("\n")
    return null
  }

  override fun visitPackage(e: PackageElement?, p: Void?): Void? {
//        debug.append(e.toString()).append("\n")
    return null
  }

}