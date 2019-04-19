package net.lab0.kuda.wrapper

import com.google.common.io.Resources
import com.squareup.kotlinpoet.ClassName
import com.squareup.kotlinpoet.CodeBlock
import com.squareup.kotlinpoet.FileSpec
import com.squareup.kotlinpoet.FunSpec
import com.squareup.kotlinpoet.KModifier
import com.squareup.kotlinpoet.ParameterSpec
import com.squareup.kotlinpoet.PropertySpec
import com.squareup.kotlinpoet.TypeName
import com.squareup.kotlinpoet.TypeSpec
import com.squareup.kotlinpoet.asClassName
import jcuda.Pointer
import jcuda.driver.CUdeviceptr
import jcuda.driver.CUfunction
import jcuda.driver.CUmodule
import jcuda.driver.JCudaDriver
import kastree.ast.Node
import kastree.ast.Visitor
import kastree.ast.psi.Parser
import net.lab0.kuda.KernelParameters
import net.lab0.kuda.KudaContext
import net.lab0.kuda.annotation.Global
import net.lab0.kuda.annotation.Return
import net.lab0.kuda.hasAnnotation
import net.lab0.kuda.withAnnotation
import javax.lang.model.element.Element
import javax.lang.model.type.TypeKind

private val Node.Decl.Func.Param.devicePointerName: String
  get() = "devicePointer_" + this.name

class CallWrapperGenerator(source: String, private val outputPackage: String) {
  private val ast = Parser.parseFile(source)

  private val fileRoot by lazy {
    lateinit var astFile: Node.File
    Visitor.visit(ast) { node, _ ->
      if (node is Node.File) {
        astFile = node
      }
    }
    astFile
  }

  private val kernel by lazy {
    // TODO assert only 1 kernel per file
    fileRoot.decls.withAnnotation("Kernel").first()
  }

  private val globalFunction by lazy {
    // TODO assert only 1 global per file?
    kernel.members
        .mapNotNull {
          it as? Node.Decl.Func
        }
        .first {
          it.hasAnnotation(Global::class)
        }
  }

  private val returnParameters by lazy {
    globalFunction.params.filter {
      it.hasAnnotation(Return::class)
    }
  }

  private val returnType: TypeName by lazy {
    when {
      returnParameters.isEmpty() -> Unit::class.asClassName()
      returnParameters.size == 1 -> returnParameters.first().type!!.asTypeName()
      else -> TODO("more than 1 return param")
    }
  }

  private val inputParameters by lazy {
    globalFunction.params.filter {
      !it.hasAnnotation(Return::class)
    }
  }

  private val allParameters = inputParameters + returnParameters

  val className by lazy {
    ClassName(outputPackage, kernel.name + "Wrapper")
  }

  val callWrapper: TypeSpec by lazy {
    TypeSpec
        .classBuilder(className)
        .superclass(KudaContext::class)
        .addProperty(
            PropertySpec
                .builder("cudaResourceName", String::class)
                .initializer("%S", outputPackage.split(".").joinToString("/", postfix = "/${kernel.name}.kt.cu"))
                .build()
        )
        .addFunction(
            FunSpec
                .builder("invoke")
                .addModifiers(KModifier.OPERATOR)
                .addParameter(
                    ParameterSpec
                        .builder("kernelParameters", KernelParameters::class)
                        .build()
                )
                .addParameters(
                    (allParameters).map {
                      ParameterSpec
                          // TODO: support default parameters? Copy them here
                          .builder(it.name, it.type!!.asTypeName())
                          .build()
                    }
                )
                .returns(returnType)
                // INIT check
                .addCode(
                    """
                      |
                      |// initialize or check that the Cuda context was initialized
                      |Companion.init()
                      |
                    """.trimMargin()
                )
                // PTX loading
                .addCode(
                    """
                      |
                      |// Load the ptx file.
                      |val ptxFileName = compileCudaToPtx(%T.getResource(cudaResourceName))
                      |val module = %T()
                      |%T.cuModuleLoad(module, ptxFileName)
                      |
                    """.trimMargin(),
                    Resources::class, // TODO: no need to depend on guava
                    CUmodule::class,
                    JCudaDriver::class
                )
                // Global function pointer
                .addCode(
                    """
                      |
                      |// Obtain a function pointer to the %S function.
                      |val function = %T()
                      |%T.cuModuleGetFunction(function, module, %S)
                      |
                    """.trimMargin(),
                    globalFunction.name.toString(),
                    CUfunction::class,
                    JCudaDriver::class,
                    globalFunction.name.toString()
                )
                // Malloc input and return
                .also { builder ->
                  allParameters
                      .filter { it.isArray() }
                      .forEach { param ->
                        builder
                            .addCode(
                                CodeBlock.of(
                                    """
                                      |
                                      |// malloc for %L
                                      |val %L = %T()
                                      |JCudaDriver.cuMemAlloc(%L, %L * %L.size.toLong())
                                      |
                                    """.trimMargin(),
                                    param.name,
                                    param.devicePointerName,
                                    CUdeviceptr::class,
                                    param.devicePointerName,
                                    param.sizeOf(),
                                    param.name
                                )
                            )
                      }
                }
                // Copy input from host to device
                .also { builder ->
                  allParameters
                      .filter { it.isArray() }
                      .forEach { param ->
                        builder
                            .addCode(
                                CodeBlock.of(
                                    """
                                      |
                                      |// Copy %L from host to device
                                      |JCudaDriver.cuMemcpyHtoD(%L, %T.to(%L), %L * %L.size.toLong())
                                      |
                                    """.trimMargin(),
                                    param.name,
                                    param.devicePointerName,
                                    CUdeviceptr::class,
                                    param.name,
                                    param.sizeOf(),
                                    param.name
                                )
                            )
                      }
                }
                // kernel parameters pointers
                .also { builder ->
                  val pointerCodeBlocks = allParameters.map { param ->
                    if (param.isArray()) {
                      CodeBlock.of("%T.to(%L)", Pointer::class, param.devicePointerName)
                    } else {
                      CodeBlock
                          .of("%T.to(%T(1){%L})", Pointer::class, param.type!!.toArrayType(), param.name)
                    }
                  }
                  builder.addCode(
                      CodeBlock.of(
                          """
                            |
                            |// send parameters to the kernel
                            |val kernelParams = %T.to(
                            |%L
                            |)
                            |
                          """.trimMargin(),
                          Pointer::class,
                          CodeBlock.of(
                              pointerCodeBlocks.joinToString(",\n") { "%L" },
                              *pointerCodeBlocks.toTypedArray()
                          )
                      )
                  )
                }
                // kernel call
                .addCode(
                    CodeBlock.of(
                        """
                          |
                          |// Call the %S function
                          |JCudaDriver.cuLaunchKernel(
                          |    function,
                          |    kernelParameters.gridDimX,
                          |    kernelParameters.gridDimY,
                          |    kernelParameters.gridDimZ,
                          |    kernelParameters.blockDimX,
                          |    kernelParameters.blockDimY,
                          |    kernelParameters.blockDimZ,
                          |    kernelParameters.sharedMemBytes,
                          |    kernelParameters.hStream,
                          |    kernelParams,
                          |    kernelParameters.extra
                          |)
                          |
                          |// wait for the computation to finish
                          |JCudaDriver.cuCtxSynchronize()
                          |
                        """.trimMargin(),
                        globalFunction.name.toString()
                    )
                )
                // Copy data from device to host
                .also { builder ->
                  returnParameters.forEach { param ->
                    if (param.isArray()) {
                      builder.addCode(
                          CodeBlock.of(
                              """
                                |
                                |// Copy %L from device to host
                                |%T.cuMemcpyDtoH(%T.to(%L), %L, %L * %L.size.toLong())
                                |
                                |
                              """.trimMargin(),
                              param.name,
                              JCudaDriver::class,
                              Pointer::class,
                              param.name,
                              param.devicePointerName,
                              param.sizeOf(),
                              param.name
                          )
                      )
                    } else {
                      TODO("Can we copy a non array element from device to host?")
                    }
                  }
                }
                // Freedom!
                .also { builder ->
                  allParameters
                      .filter { it.isArray() }
                      .forEach { param ->
                        builder
                            .addCode(
                                CodeBlock.of(
                                    """
                                      |
                                      |// free for %L
                                      |JCudaDriver.cuMemFree(%L)
                                      |
                                    """.trimMargin(),
                                    param.name,
                                    param.devicePointerName
                                )
                            )
                      }
                }
                .addCode(
                    when (returnParameters.size) {
                      0 -> CodeBlock.of("")
                      1 -> CodeBlock.of(
                          """
                            |
                            |// Job's finished ! :)
                            |return %L
                            |
                          """.trimMargin(),
                          returnParameters.first().name
                      )
                      else -> TODO("More than 1 return type, 'all crews reporting'")
                    }
                )
                .build()
        )
        .build()
  }

  val callWrapperFile by lazy {
    FileSpec.get(className.packageName, callWrapper)
  }
}