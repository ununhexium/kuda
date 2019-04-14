package net.lab0.kuda.wrapper

import com.google.common.io.Resources
import com.squareup.kotlinpoet.ClassName
import com.squareup.kotlinpoet.CodeBlock
import com.squareup.kotlinpoet.FileSpec
import com.squareup.kotlinpoet.FunSpec
import com.squareup.kotlinpoet.KModifier
import com.squareup.kotlinpoet.ParameterSpec
import com.squareup.kotlinpoet.PropertySpec
import com.squareup.kotlinpoet.TypeSpec
import com.squareup.kotlinpoet.asClassName
import com.squareup.kotlinpoet.asTypeName
import jcuda.Pointer
import jcuda.Sizeof
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
import net.lab0.kuda.exception.CantConvert
import net.lab0.kuda.hasAnnotation
import net.lab0.kuda.withAnnotation
import javax.lang.model.element.Element
import javax.lang.model.type.ArrayType
import javax.lang.model.type.TypeKind
import kotlin.reflect.KClass

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

  private val returnType: ClassName by lazy {
    when {
      returnParameters.isEmpty() -> Unit::class.asClassName()
      returnParameters.size == 1 -> returnParameters.first().type!!.asKClass().asClassName()
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
                          .builder(it.name, it.type!!.asKClass().asTypeName())
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
                      |val ptxFileName = compileCudaToPtx(%T.getResource(cudaResourceName))
                      |
                    """.trimMargin(),
                    Resources::class // TODO: no need to depend on guava
                )
                .addCode(
                    """
                      |
                      |// Load the ptx file.
                      |val module = %T()
                      |%T.cuModuleLoad(module, ptxFileName)
                      |
                    """.trimMargin(),
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
                  inputParameters
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
                          .of("%T.to(%T(1){%L})", Pointer::class, param.type!!.asKClass().toArrayType(), param.name)
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
                .also {builder ->
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

  private val arrayWrap: (Element, String) -> String = { element, name ->
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
}

private fun KClass<*>.toArrayType(): KClass<*> {
  return when (this) {
    Byte::class -> ByteArray::class
    UByte::class -> UByteArray::class
    Int::class -> IntArray::class
    UInt::class -> UIntArray::class
    Long::class -> LongArray::class
    ULong::class -> ULongArray::class
    Short::class -> ShortArray::class
    UShort::class -> UShortArray::class

    Char::class -> CharArray::class
    Double::class -> DoubleArray::class
    Float::class -> FloatArray::class

    else -> throw CantConvert("Can't get size of $this")
  }
}

private fun Node.Decl.Func.Param.isArray(): Boolean {
  // TODO: in what case is there more than 1 piece? Is that for classes generics?
  val ref = this.type!!.ref as Node.TypeRef.Simple
  val firstPiece = ref.pieces.first()
  return firstPiece.name.matches(Regex("(U?(Byte|Int|Long|Short)|Char|Float|Double)Array"))
}

private fun Node.Decl.Func.Param.sizeOf() = findSizeOf(this)

private fun Node.Type.asKClass(): KClass<*> {
  val simple = this.ref as? Node.TypeRef.Simple
      ?: throw IllegalArgumentException("Can't extract classname from $this")

  // TODO: can there be more than 1 piece?
  if (simple.pieces.size > 1) throw IllegalArgumentException("Don't know how to handle multiple pieces $this")
  val firstPiece = simple.pieces.first()

  return when (firstPiece.name) {
    "Float" -> Float::class
    "Int" -> Int::class
    "FloatArray" -> FloatArray::class
    "IntArray" -> IntArray::class
    else -> TODO("Convert type $firstPiece")
    // TODO: Doesnt work :( else -> Class.forName(firstPiece.name).kotlin.asClassName()
  }
}

private fun findSizeOf(param: Node.Decl.Func.Param): CodeBlock {
  return when (param.type!!.asKClass()) {
    // TODO: int test for each of them
    ByteArray::class -> CodeBlock.of("%T.BYTE", Sizeof::class)
    UByteArray::class -> CodeBlock.of("%T.BYTE", Sizeof::class)
    IntArray::class -> CodeBlock.of("%T.INT", Sizeof::class)
    UIntArray::class -> CodeBlock.of("%T.INT", Sizeof::class)
    LongArray::class -> CodeBlock.of("%T.LONG", Sizeof::class)
    ULongArray::class -> CodeBlock.of("%T.LONG", Sizeof::class)
    ShortArray::class -> CodeBlock.of("%T.SHORT", Sizeof::class)
    UShortArray::class -> CodeBlock.of("%T.SHORT", Sizeof::class)

    CharArray::class -> CodeBlock.of("%T.CHAR", Sizeof::class)
    DoubleArray::class -> CodeBlock.of("%T.DOUBLE", Sizeof::class)
    FloatArray::class -> CodeBlock.of("%T.FLOAT", Sizeof::class)
    else -> throw CantConvert("Can't get size of $param")
  }
}