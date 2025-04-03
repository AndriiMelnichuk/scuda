package scuda.tensor

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

