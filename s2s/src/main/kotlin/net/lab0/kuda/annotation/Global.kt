package net.lab0.kuda.annotation

/**
 * Equivalent of the __global__ macro in Cuda.
 *
 * This in the function the transpiler is going to look
 * for when converting a class annotated with [Kernel] to a Cuda file.
 */
@Target(AnnotationTarget.FUNCTION)
annotation class Global
