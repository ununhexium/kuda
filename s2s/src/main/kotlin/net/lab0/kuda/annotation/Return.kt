package net.lab0.kuda.annotation

/**
 * Global function parameters annotated with [Return] will not copy the given data to the device,
 * thus saving some overhead. The data will copied from the device to the host after computation.
 */

@Target(AnnotationTarget.VALUE_PARAMETER)
annotation class Return
