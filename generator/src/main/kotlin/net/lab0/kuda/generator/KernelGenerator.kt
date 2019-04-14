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
import net.lab0.kuda.annotation.Return
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
      val globalFunctions = it.enclosedElements.filter { e ->
        e.kind == ElementKind.METHOD && e.getAnnotation(Global::class.java) != null
      }

      if (globalFunctions.isEmpty()) {
        throw IllegalSourceCode("Didn't find any global function")
      }

      if (globalFunctions.size > 1) {
        throw IllegalSourceCode("Only 1 global function per source file is supported")
      }

      val globalFunction = globalFunctions.first()

      val myVisitor = MyVisitor()
      globalFunction.accept(myVisitor, null)

      val className = it.simpleName.toString()
      generateClass(className, "net.lab0.kuda.generated.kernel", myVisitor.executables.first())
    }

    return true
  }

  override fun getSupportedAnnotationTypes(): MutableSet<String> {
    println("getSupportedAnnotationTypes")
    return mutableSetOf(Kernel::class.java.name)
  }

  private fun generateClass(className: String, pack: String, globalFunction: ExecutableElement) {
    // TODO: for return only elements, don't copy from host to device
    val returned = globalFunction.parameters.filter {
      it.getAnnotation(Return::class.java) != null
    }

    val file = FileSpec.builder(pack, className + "Generated")
        .addType(
            TypeSpec.classBuilder(className + "Generated")
                .superclass(KudaContext::class)
                .addProperty(
                    PropertySpec
                        .builder("cudaResourceName", String::class)
                        .initializer("%S", "net/lab0/kuda/kernel/$className.kt.cu")
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
                                  |init()
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
                        .also { function ->
                          copyDataFromDeviceToHost(returned, function)
                        }
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
          TypeKind.DOUBLE -> DoubleArray::class.asTypeName()
          TypeKind.LONG -> LongArray::class.asTypeName()
          TypeKind.FLOAT -> FloatArray::class.asTypeName()
          // TODO: more types
          else -> throw IllegalStateException("The kind ${type.componentType.kind} is not supported")
        }
      }
      TypeKind.INT -> Int::class.asTypeName()
      TypeKind.LONG -> Long::class.asTypeName()
      // TODO: more
      else -> throw IllegalStateException("The kind ${it.asType().kind} is not supported")
    }
  }

  private fun copyDataFromDeviceToHost(returned: List<VariableElement>, function: FunSpec.Builder) {
    returned.forEach { param ->
      if (param.isArray()) {
        val sizeOf = findSizeOf(param.asType() as ArrayType)

        function.addCode(
            CodeBlock.of(
                """
                  |// copy data back to host
                  |JCudaDriver.cuMemcpyDtoH(Pointer.to(%L), devicePointer_%L, %LL * %L.size)
                  |
                  |
                """.trimMargin(),
                param.simpleName,
                param.simpleName,
                sizeOf,
                param.simpleName
            )
        )
      } else {
        TODO("Can we return a non array element?")
      }
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
                  |JCudaDriver.cuMemFree(devicePointer_%L)
                  |
                  |
                """.trimMargin(),
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
      TypeKind.DOUBLE -> Sizeof.DOUBLE
      TypeKind.LONG -> Sizeof.LONG
      // TODO: more types
      else -> throw IllegalStateException("The kind ${param.componentType.kind} is not supported")
    }
  }

  private fun generateKernelParameters(params: List<VariableElement>): CodeBlock {
    val arrayWrap: (Element, String) -> String = { element, name ->
      val kind = element.asType().kind
      when (kind) {
        TypeKind.BOOLEAN -> "BooleanArray(1){$name}"
        TypeKind.INT -> "IntArray(1){$name}"
        TypeKind.LONG -> "LongArray(1){$name}"
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
              arrayWrap(it, "devicePointer_" + it.simpleName.toString()) +
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

