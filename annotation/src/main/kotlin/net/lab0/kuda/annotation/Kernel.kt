package net.lab0.kuda.annotation

import java.lang.annotation.ElementType
import java.lang.annotation.RetentionPolicy

/**
 * Classe annotated with this [Kernel] will be looked for at
 * Cuda file generation and annotation processing.
 */

@Target(AnnotationTarget.CLASS)
@Retention(AnnotationRetention.RUNTIME)
@MustBeDocumented
annotation class Kernel
