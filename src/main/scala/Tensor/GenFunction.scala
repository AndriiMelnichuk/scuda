package scuda.Tensor

import jcuda.Pointer
import jcuda.runtime._
import jcuda.runtime.JCuda._
import jcuda.Sizeof
import jcuda.jcublas.*
import jcuda.driver.CUmodule
import jcuda.driver.CUfunction
import jcuda.driver.JCudaDriver

trait GeneralFunction:
	lazy val args: Seq[Tensor]
	lazy val forward: Storage
	def backward(argument: Tensor, chainGrad: Storage): Storage
	def elementalBackward(chainGrad: Storage): Seq[Storage] = 
		args.map(backward(_, chainGrad))

trait ReplicatibleFunction:
	def apply(x: Tensor): Tensor
	def replicate(grad: Map[Tensor, Storage], opt: Optimizer): ReplicatibleFunction

class ForwardLayer(val w: Tensor, val b: Tensor) extends ReplicatibleFunction:
	def apply(x: Tensor): Tensor = 
		val be = heightExpander(b, x.storage.shape(0))
		val res = x ** (w.T) + be
		new Tensor(new GeneralFunction {
			lazy val args: Seq[Tensor] = res.origin.args
			lazy val forward = res.storage
			def backward(arg: Tensor, chainGrad: Storage) = res.origin.backward(arg, chainGrad)
				
		}, x.hasVar || b.hasVar || w.hasVar)

	def replicate(grad: Map[Tensor, Storage], opt: Optimizer): ForwardLayer = 
		val newW = opt(w.storage, grad(w))
		val newB = opt(b.storage, grad(b))
		new ForwardLayer(Tensor(newW, w.hasVar), Tensor(newB, w.hasVar))

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


def heightExpander(x: Tensor, n: Int): Tensor =
	val host = x.storage match
		case _: CudaStorage  => "cuda"
		case _: ArrayStorage => "cpu"
		
	val ones = Tensor(Storage.ones(Seq(n, 1), host), false)

	ones ** (x.T)

	