package net.lab0.kuda.generator

import com.google.auto.service.AutoService
import com.google.common.io.Resources
import com.squareup.kotlinpoet.ClassName
import com.squareup.kotlinpoet.CodeBlock
import com.squareup.kotlinpoet.FileSpec
import com.squareup.kotlinpoet.FunSpec
import com.squareup.kotlinpoet.KModifier
import com.squareup.kotlinpoet.ParameterSpec
import com.squareup.kotlinpoet.PropertySpec
import com.squareup.kotlinpoet.TypeSpec
import com.squareup.kotlinpoet.asTypeName
import jcuda.Pointer
import jcuda.Sizeof
import jcuda.driver.CUdeviceptr
import jcuda.driver.CUfunction
import jcuda.driver.CUmodule
import jcuda.driver.JCudaDriver
import net.lab0.kuda.KernelParameters
import net.lab0.kuda.KudaContext
import net.lab0.kuda.annotation.Global
import net.lab0.kuda.annotation.Kernel
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import java.io.File
import javax.annotation.processing.*
import javax.lang.model.SourceVersion
import javax.lang.model.element.TypeElement
import javax.lang.model.element.Element
import javax.lang.model.element.ElementKind
import javax.lang.model.element.ExecutableElement
import javax.lang.model.element.VariableElement
import javax.lang.model.type.ArrayType
import javax.lang.model.type.TypeKind


@AutoService(Processor::class)
@SupportedSourceVersion(SourceVersion.RELEASE_8)
class KernelGenerator : AbstractProcessor() {
  companion object {
    private val log: Logger by lazy { LoggerFactory.getLogger(this::class.java.name) }
    const val KAPT_KOTLIN_GENERATED_OPTION_NAME = "kapt.kotlin.generated"
  }

  override fun process(set: MutableSet<out TypeElement>, roundEnv: RoundEnvironment): Boolean {
    roundEnv.getElementsAnnotatedWith(Kernel::class.java).forEach {
      val globalFunction = it.enclosedElements.first { e ->
        e.kind == ElementKind.METHOD && e.getAnnotation(Global::class.java) != null
      }
      val executableVisitor = MyVisitor()
      globalFunction.accept(executableVisitor, null)

      // TODO: if globals count != 1 -> error
      val className = it.simpleName.toString()
      generateClass(className, "net.lab0.kuda.generated.kernel", executableVisitor.executables.first())
    }

    return true
  }

  override fun getSupportedAnnotationTypes(): MutableSet<String> {
    println("getSupportedAnnotationTypes")
    return mutableSetOf(Kernel::class.java.name)
  }

  private fun generateClass(className: String, pack: String, globalFunction: ExecutableElement) {
    val returned = listOf<Any>()

    val file = FileSpec.builder(pack, className + "Generated")
        .addType(
            TypeSpec.classBuilder(className + "Generated")
                .superclass(KudaContext::class)
                .addProperty(
                    PropertySpec
                        .builder("cudaResourceName", String::class)
                        .initializer("%S", "net/lab0/kuda/kernel/$className.cuda")
                        .build()
                )
                .addFunction(
                    FunSpec
                        .builder("invoke")
                        .addModifiers(KModifier.OPERATOR)
                        .addParameter(
                            ParameterSpec
                                .builder("kernelParams", KernelParameters::class)
                                .build()
                        )
                        .addParameters(
                            globalFunction.parameters.map {
                              ParameterSpec
                                  .builder(it.simpleName.toString(), getKotlinType(it))
                                  .build()
                            }
                        )
                        .returns(Unit::class)
                        .addCode(
                            CodeBlock.of(
                                """
                                  |val ptxFileName = compileCudaToPtx(%T.getResource(cudaResourceName))
                                  |
                                  |// Load the ptx file.
                                  |val module = %T()
                                  |%T.cuModuleLoad(module, ptxFileName)
                                  |
                                  |// Obtain a function pointer to the function.
                                  |val function = %T()
                                  |%T.cuModuleGetFunction(function, module, %S)
                                  |
                                  |
                                """.trimMargin(),
                                Resources::class,
                                CUmodule::class,
                                JCudaDriver::class,
                                CUfunction::class,
                                JCudaDriver::class,
                                globalFunction.simpleName.toString()
                            )
                        )
                        .also { function ->
                          generateMemoryAllocators(globalFunction, function)
                        }
                        // TODO: returned values malloc
                        .addCode(
                            generateKernelParameters(globalFunction.parameters)
                        )
                        .addCode(
                            CodeBlock.of(
                                """
                                  |// Call the kernel function.
                                  |JCudaDriver.cuLaunchKernel(
                                  |    function,
                                  |    kernelParams.gridDimX,
                                  |    kernelParams.gridDimY,
                                  |    kernelParams.gridDimZ,
                                  |    kernelParams.blockDimX,
                                  |    kernelParams.blockDimY,
                                  |    kernelParams.blockDimZ,
                                  |    kernelParams.sharedMemBytes,
                                  |    kernelParams.hStream,
                                  |    kernelParameters,
                                  |    kernelParams.extra
                                  |)
                                  |JCudaDriver.cuCtxSynchronize()
                                  |
                                  |
                                """.trimMargin()
                            )
                        )
                        // TODO: copy returned data
                        .also { function ->
                          generateMemoryDeallocators(globalFunction, function)
                        }
                        .build()
                )
                .build()
        )
        .build()

    val kaptKotlinGeneratedDir = processingEnv.options[KAPT_KOTLIN_GENERATED_OPTION_NAME]
    file.writeTo(File(kaptKotlinGeneratedDir, "$className.kt"))
  }

  private fun getKotlinType(it: VariableElement): ClassName {
    return when (it.asType().kind) {
      TypeKind.ARRAY -> {
        val type = it.asType()
        type as ArrayType
        when (type.componentType.kind) {
          TypeKind.INT -> IntArray::class.asTypeName()
          else -> throw IllegalStateException("The kind ${type.componentType.kind} is not supported")
        }
      }
      TypeKind.INT -> Int::class.asTypeName()
      else -> throw IllegalStateException("The kind ${it.asType().kind} is not supported")
    }
  }

  private fun generateMemoryAllocators(
      globalFunction: ExecutableElement,
      function: FunSpec.Builder
  ) {
    globalFunction.parameters.forEach { param ->
      if (param.isArray()) {
        val sizeOf = findSizeOf(param.asType() as ArrayType)

        function.addCode(
            CodeBlock.of(
                """
                  |// malloc for %L
                  |val devicePointer_%L = %T()
                  |JCudaDriver.cuMemAlloc(devicePointer_%L, %LL * %L.size)
                  |JCudaDriver.cuMemcpyHtoD(devicePointer_%L, Pointer.to(%L), %LL * %L.size)
                  |
                  |
                """.trimMargin(),
                param.simpleName,
                param.simpleName,
                CUdeviceptr::class,
                param.simpleName,
                sizeOf,
                param.simpleName,
                param.simpleName,
                param.simpleName,
                sizeOf,
                param.simpleName
            )
        )
      }
    }
  }

  private fun generateMemoryDeallocators(globalFunction: ExecutableElement, function: FunSpec.Builder) {
    globalFunction.parameters.forEach { param ->
      if (param.isArray()) {
        val sizeOf = findSizeOf(param.asType() as ArrayType)

        function.addCode(
            CodeBlock.of(
                """
                  |// free %L
                  |JCudaDriver.cuMemcpyDtoH(
                  |    Pointer.to(%L),
                  |    devicePointer_%L,
                  |    %LL * %L.size
                  |)
                  |JCudaDriver.cuMemFree(devicePointer_%L)
                  |
                  |
                """.trimMargin(),
                param.simpleName,
                param.simpleName,
                param.simpleName,
                sizeOf,
                param.simpleName,
                param.simpleName
            )
        )
      }
    }
  }

  private fun findSizeOf(param: ArrayType): Int {
    return when (param.componentType.kind) {
      TypeKind.INT -> Sizeof.INT // TODO: use literal instead of copying the value
      else -> throw IllegalStateException("The kind ${param.componentType.kind} is not supported")
    }
  }

  private fun generateKernelParameters(params: List<VariableElement>): CodeBlock {
    val arrayWrap: (Element, String) -> String = { element, name ->
      val kind = element.asType().kind
      when (kind) {
        TypeKind.BOOLEAN -> "BooleanArray(1){$name}"
        TypeKind.INT -> "IntArray(1){$name}"
        TypeKind.ARRAY -> name
        // TODO add other types
        else -> throw IllegalStateException("The kind $kind is not supported")
      }
    }
    return CodeBlock.of(
        """
          |val kernelParameters = %T.to(
          |${
        params.joinToString(",\n") {
          "    %T.to(" +
              arrayWrap(it, it.simpleName.toString()) +
              ")"
        }
        }
          |)
          |
          |
        """.trimMargin(),
        *Array(params.size + 1) { Pointer::class }
    )
  }

  fun VariableElement.isArray(): Boolean {
    return this.asType().kind == TypeKind.ARRAY
  }
}

