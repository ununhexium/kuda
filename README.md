# Kuda

Write a kernel in Kotlin, run some magic and voilà.
A freshly baked kernel wrapper ready to be executed.


This is a proof of concept to provide a way to write Cuda kernels in Kotlin.
The Kotlin code is transpiled from Kotlin into CPP/Cuda source code.
That code is then compiled to a `ptx` file with `nvcc` to be executed. 


## Hello Kuda

First, define a kernel class with the `@Kernel` annotation.
Only supports 1 `@Kernel` per file.

Declare a global function with the `@Global` annotation.
Only supports 1 `@Global` per kernel.

```kotlin
@Kernel
class SaxpySample: KudaContext() {
  @Global
  fun saxpy(n: Int, a: Float, x: FloatArray, @Return y: FloatArray) {
    val i: Int = blockIdx.x * blockDim.x + threadIdx.x
    if (i < n) y[i] = a * x[i] + y[i]
  }
}
```

Run some generator magic.
This is currently achieved as a gradle task.
Have a look at the `sample` project, 
[`kuda`](https://github.com/ununhexium/kuda/blob/wip/sample/build.gradle.kts) 
task for details.

After code generation, a kernel call wrapper is available.

```kotlin
fun main() {
  val saxpy = SaxpySampleWrapper()
  val a = 0.5f
  val x = FloatArray(116) { it.toFloat() }
  val y = FloatArray(116) { -1.0f }
  
  val res = saxpy(KernelParameters.for1D(x.size), 10, a, x, y)
  println(res.joinToString())
}
```

All the boilerplate code is in the wrapper.
This uses [jCuda](http://www.jcuda.de/)
to forward the kernel call to the graphics card.

-------------------------------

## Features

#⚠
**Very experimental.**

Currently supports some basic C-like operations with a lot of restrictions.


### [Data types](s2s/src/test/kotlin/net/lab0/kuda/sample/DataTypeKernel.kt)

Kotlin data types are mapped to their C equivalent according to the following table.
 
 
| Kotlin       | C++                   |
|-------------:|:----------------------|
#| Boolean      | Not supported[^1]     |
| Byte         | char                  |
#| UByte        | unsigned char         |
| Short        | short                 |
#| UShort       | unsigned short        |
| Int          | int                   |
#| UInt         | unsigned int          |
| Long         | long                  |
#| ULong        | unsigned long         |
|              |                       |
| Float        | float                 |
| Double       | double                |
|              |                       |
| BooleanArray | Not supported[^1]     |
| ByteArray    | char *                |
#| UByteArray   | unsigned char *       |
| ShortArray   | short *               |
#| UShortArray  | unsigned short *      |
| IntArray     | int *                 |
#| UIntArray    | unsigned int *        |
| LongArray    | long *                |
#| ULongArray   | unsigned long *       |
|              |                       |
| FloatArray   | float *               |
| DoubleArray  | double *              |

Uses kotlin 1.3 experimental unsigned types.

[^1]: The size of a Boolean is JVM [implementation dependant](https://docs.oracle.com/javase/tutorial/java/nutsandbolts/datatypes.html) 
and JCuda doesn't offer a way to get a boolean's size nor pointer to a boolean array.
As a workaround, use any of the integer types.

### [Cast](s2s/src/test/kotlin/net/lab0/kuda/sample/PrimitivesCastKernel.kt)

Casts are supported with the `variable.toXxx()` 
kotlin cast notation for all primitives types 
expect between float and unsigned types as Kotlin 
doesn't propose it.

### [Operators](s2s/src/test/kotlin/net/lab0/kuda/sample/OperatorsKernel.kt)

Tested operators are

Arithmetic
* `+`
* `+` unary
* `-`
* `-` unary
* `*` multiplication
* `/`
* `%`
* `++` prefixed
* `--` prefixed
* `++` postfixed
* `--` postfixed

Relational
* `(` `)` priority, not function call 
* `>` 
* `<` 
* `>=` 
* `<=` 

Logical
* `&&`
* `||`
* `!`

Binary
* `&`
* `|`
* `^`

| Kotlin | C++  |
|-------:|:-----|
|and     |  &   |
|or      |&#124;|
|xor     | ^    |

### [Control structures](s2s/src/test/kotlin/net/lab0/kuda/sample/ControlKernel.kt)

* `while`
* `if`

`for` is explicitly not supported as the syntax are very different.
While will do the job just fine.


### [Matrices](s2s/src/test/kotlin/net/lab0/kuda/sample/Matrix2DKernel.kt)

Supports C matrix notation `int [][] foo` and `int ** foo` 
with nested arrays: `val foo: Array<IntArray>` only inside the kernel.
Passing such arguments via the wrapper is not supported.


-------------------------------

## Limitations

A lot... ʘ︵ʘ

### No chars translation

No conversion for `Char`s and `CharArray`s.

### Variable names are completely unchecked.

Don't use a valid variable kotlin name which is a C++ keyword,
such as `extern`, `bool`, `unsigned`, ...

Names are not resolved. Use `threadIdx.x`, not `KudaContext.threadIdx.x`.

Kotlin types are converted by name (java.lang.Class.getSimpleName).
Using a class named BoolArray will make it translated to 
`bool *` whichever package it comes from.

### No type inference

`val b = true` will not work. Use explicit types `val b:Boolean = true`

### For loop

No support for `for(x in xs) { ... }`

### Cuda functions calls

None so far.

### Operators

Not tested

### Function parameters

Kotlin forbids the reassignment of function parameters.
Either redeclare the variable, or use a 1 element array.

### Function calls

Limited to binary operators special cases


-------------------------------

## TODOs

Lots of them in the code ! This is a section for TODOs which are not bound to a specific code location

### More samples

Grab all the nVidia doc and try their samples

### Gradle plugin

Formalize the code generator `kuda` task as gradle plugin.

### OSX/Windows adaptations

Especially check for paths validity. The rest should be handled by the libs.

### Cuda headers and function import

Propose placeholders for all the Cuda functions.

### Data classes

Map C struct to kotlin data classes

### Cast

```kotlin
val i:Long = 1
val j:Int = i.toInt()
```

Should be translated as
```c++
long i = 1
int j = (int) i
```

Same for `toFloat`, `toDouble`, ...

-------------------------------

### Automatic initialization

In Kotlin/Java int, double... have initial values.
Also init these values in C.

### Multidimensional array passing via wrapper

# Alternatives


## Cuda

### JCuda

Write your kernels in C, with the true Cuda API and call them from the JVM.


## OpenCL

### Aparapi

Kuda is not translating from bytecode to cuda.
It's source to source.

For a bytecode to kernel approach, you may have a look at 
[aparapi](https://github.com/aparapi/aparapi) 
which provides such a mechanism for OpenCL.

### JavaCL

https://github.com/nativelibs4java/JavaCL

### JogAmp

http://jogamp.org/jocl/www/

### LWJGL openGL wrappers

https://www.lwjgl.org/

