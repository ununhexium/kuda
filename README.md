# Kuda

Write a kernel in Kotlin, run some magic and voilà.
A freshly baked kernel wrapper ready to be executed.


This is a proof of concept to provide a way to write Cuda kernels in Kotlin.
The Kotlin code is translated from Kotlin source code into CPP source code.
That code is then compile to a `ptx` with nvcc. 


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

**⚠ Very experimental.**

Currently supports some basic C-like operations with a lot of restrictions.

### Data types

Kotlin data types are mapped to their C equivalent according to the following table.
 
 
| Kotlin       | C               |
|-------------:|:----------------|
| Boolean      | bool            |
| Byte         | char            |
| UByte        | unsigned char   |
| Short        | short           |
| UShort       | unsigned short  |
| Int          | int             |
| UInt         | unsigned int    |
| Long         | long            |
| ULong        | unsigned long   |
|              |                 |
| Float        | float           |
| Double       | double          |
|              |                 |
| BooleanArray | bool *          |
| ByteArray    | char *          |
| UByteArray   | unsigned char * |
| ShortArray   | short *         |
| UShortArray  | unsigned short *|
| IntArray     | int *           |
| UIntArray    | unsigned int *  |
| LongArray    | long *          |
| ULongArray   | unsigned long * |
|              |                 |
| FloatArray   | float *         |
| DoubleArray  | double *        |

Uses kotlin 1.3 experimental unsigned types.

### Matrices

Supports


No conversion for `Char`s.

-------------------------------

## Limitations

A lot... ʘ︵ʘ


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

-------------------------------

# Alternatives


## Cuda

### JCuda

Write your kernels in C, with the true Cuda API and call them from the JVM.


## OpenCL

### Aparapi

This is not translating from bytecode to cuda.

For such an approach, you may have a look at 
[aparapi](https://github.com/aparapi/aparapi) 
which provides such a mechanism for OpenCL.

### JavaCL

https://github.com/nativelibs4java/JavaCL

### JogAmp

http://jogamp.org/jocl/www/

### LWJGL open LC wrappers

https://www.lwjgl.org/
