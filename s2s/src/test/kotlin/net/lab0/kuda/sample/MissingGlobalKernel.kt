package net.lab0.kuda.sample

import net.lab0.kuda.annotation.Kernel

/**
 * This is not a mistake.
 *
 * It's there to test exception generation when there
 * is no [Global] annotation in the kernel.
 */
@Kernel
class MissingGlobalKernel
