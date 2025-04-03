package scuda.tensor.ai

import scuda.tensor. { Tensor, Storage, GeneralFunction }

import scuda.tensor.cpu.ArrayStorage
import scuda.tensor.cuda.CudaStorage

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


class ReLU() extends ReplicatibleFunction:
	def apply(x: Tensor): Tensor = ???
	def replicate(grad: Map[Tensor, Storage], opt: Optimizer): ReplicatibleFunction = ???

// TODO change to device match
def heightExpander(x: Tensor, n: Int): Tensor =
	val host = x.storage match
		case _: CudaStorage  => "cuda"
		case _: ArrayStorage => "cpu"
		
	val ones = Tensor(Storage.ones(Seq(n, 1), host), false)

	ones ** (x.T)

	