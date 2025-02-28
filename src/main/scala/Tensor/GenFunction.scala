package scuda.Tensor

import jcuda.Pointer
import jcuda.runtime._
import jcuda.runtime.JCuda._
import jcuda.Sizeof
import jcuda.jcublas.*
import jcuda.driver.CUmodule
import jcuda.driver.CUfunction
import jcuda.driver.JCudaDriver

// class GeneralFunction(
// 	arguments: => Seq[Tensor], 
// 	forwardFun: => Storage, 
// 	backwardFun: => Seq[() => Storage],
// 	reducerFun: => Seq[Storage => Storage]
// ):
// 	def args: Seq[Tensor] = arguments
// 	def forward: Storage = forwardFun
// 	def backward: Seq[() => Storage] = backwardFun
// 	def reducer: Seq[Storage => Storage] = reducerFun

trait GeneralFunction:
	lazy val args: Seq[Tensor]
	lazy val forward: Storage
	def backward(argument: Tensor, chainGrad: Storage): Storage
	def elementalBackward(chainGrad: Storage): Seq[Storage] = 
		args.map(backward(_, chainGrad))
		 

// Todo: not working
class  SinTensor(x: Tensor) extends GeneralFunction:
	lazy val args: Seq[Tensor] = Seq(x)
	lazy val forward: Storage = 
		x.storage match
			case v: ArrayStorage => new ArrayStorage(v.storage.map(x => math.sin(x.toDouble).toFloat), v.shape)
			case v: CudaStorage => 
				val resPointer = new Pointer()
				cudaMalloc(resPointer, v.shape.product * Sizeof.FLOAT)
				val kernelSin = """
        extern "C"
        __global__ void computeSin(float *input, float *sinOut, int n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                sinOut[i] = sinf(input[i]);
            }
        }
				"""
				val module = CUmodule()
				val function = CUfunction()
				JCudaDriver.cuModuleLoadData(module, kernelSin)
				val kernelPar = Pointer.to(
					v.storage,
					resPointer,
					Pointer.to(Array(v.shape.product))
				)
				val blockSize = 32;
				val gridSize = (v.shape.product + blockSize - 1) / blockSize;
				JCudaDriver.cuLaunchKernel(function, gridSize, 1, 1, blockSize, 1, 1, 0, null, kernelPar, null);

				new CudaStorage(resPointer, v.shape)

				// val kernelCos = """
        // extern "C"
        // __global__ void computeSinCos(float *input, float *cosOut, int n) {
        //     int i = blockIdx.x * blockDim.x + threadIdx.x;
        //     if (i < n) {
        //         cosOut[i] = cosf(input[i]);
        //     }
        // }
				// """
				
		
	def backward(argument: Tensor, chainGrad: Storage): Storage = ???


