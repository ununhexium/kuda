package net.lab0.kuda.generator

import com.google.auto.service.AutoService
import com.squareup.kotlinpoet.FileSpec
import com.squareup.kotlinpoet.FunSpec
import com.squareup.kotlinpoet.TypeSpec
import java.io.File
import javax.annotation.processing.*
import javax.lang.model.SourceVersion
import javax.lang.model.element.TypeElement
import javax.annotation.processing.ProcessingEnvironment


@AutoService(Processor::class)
@SupportedSourceVersion(SourceVersion.RELEASE_8)
class KernelGenerator : AbstractProcessor() {

  override fun init(pe: ProcessingEnvironment) {
    super.init(pe)
    this.trees = Trees.instance(pe)
  }

  override fun process(set: MutableSet<out TypeElement>, roundEnv: RoundEnvironment): Boolean {
    roundEnv.getElementsAnnotatedWith(Kernel::class.java).forEach {
      it.
    }
    generateClass("TheFoo", "foo.bar")

    return true
  }

  override fun getSupportedAnnotationTypes(): MutableSet<String> {
    println("getSupportedAnnotationTypes")
    return mutableSetOf(Kernel::class.java.name)
  }

  private fun generateClass(className: String, pack: String) {
    val fileName = "Generated_$className"
    val file = FileSpec.builder(pack, fileName)
        .addType(
            TypeSpec.classBuilder(fileName)
                .addFunction(
                    FunSpec.builder("getName")
                        .addStatement("return \"World\"")
                        .build()
                )
                .build()
        )
        .build()

    val kaptKotlinGeneratedDir = processingEnv.options[KAPT_KOTLIN_GENERATED_OPTION_NAME]
    file.writeTo(File(kaptKotlinGeneratedDir, "$fileName.kt"))
  }

  companion object {
    const val KAPT_KOTLIN_GENERATED_OPTION_NAME = "kapt.kotlin.generated"
  }
}
