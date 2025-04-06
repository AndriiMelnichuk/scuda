package scuda.tensor.cuda

import scuda.tensor.Storage

import jcuda.Pointer
import jcuda.Sizeof
import jcuda.runtime.JCuda._
import jcuda.driver.JCudaDriver._
import jcuda.runtime.cudaMemcpyKind
import jcuda.driver.CUmodule
import jcuda.driver.CUfunction

object CudaOperations:
	/**
	 * Runs cuda kernels implementing element-by-element operations on 2 tensors
	 *
	 * @param a CudaStorage 1
	 * @param b CudaStorage 2
	 * @param ptxFile path to compiled cuda file
	 * @param opName name of operation
   * @param mBlockSize max count of threads in block
	 * @return operation result on CudaStorage
	 */
  def elementwiseCernelExecuter(a: CudaStorage, b: CudaStorage, ptxFile: String, opName: String, mBlockSize: Int = 1024): Storage =    
    val d_a = a.storage
    val d_b = b.storage
    val d_c = Pointer()

    cudaMalloc(d_c, a.shape.product * Sizeof.FLOAT)

    // module load
    val module = CUmodule()
    cuModuleLoad(module, ptxFile)
    val function = CUfunction()
    cuModuleGetFunction(function, module, opName)

    val kernelParams = Pointer.to(
      Pointer.to(Array(a.shape.product)),
      Pointer.to(d_a),
      Pointer.to(d_b),
      Pointer.to(d_c)
    )
    val blockSize = if (mBlockSize > a.shape.product) a.shape.product else mBlockSize
    val gridSize = (a.shape.product + mBlockSize - 1) / mBlockSize
    cuLaunchKernel(function, gridSize, 1, 1, 
                            blockSize, 1, 1, 
                            0, null, kernelParams, null);
    new CudaStorage(d_c, a.shape)

  def elementwiseScalarCernelExecuter(a: CudaStorage, b: Float, ptxFile: String, opName: String, mBlockSize: Int = 1024): Storage =    
    val d_a = a.storage
    val d_c = Pointer()

    cudaMalloc(d_c, a.shape.product * Sizeof.FLOAT)

    // module load
    val module = CUmodule()
    cuModuleLoad(module, ptxFile)
    val function = CUfunction()
    cuModuleGetFunction(function, module, opName)

    val kernelParams = Pointer.to(
      Pointer.to(Array(a.shape.product)),
      Pointer.to(d_a),
      Pointer.to(Array(b)),
      Pointer.to(d_c)
    )
    val blockSize = if (mBlockSize > a.shape.product) a.shape.product else mBlockSize
    val gridSize = (a.shape.product + mBlockSize - 1) / mBlockSize
    cuLaunchKernel(function, gridSize, 1, 1, 
                            blockSize, 1, 1, 
                            0, null, kernelParams, null);
    new CudaStorage(d_c, a.shape)