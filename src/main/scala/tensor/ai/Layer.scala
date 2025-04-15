package scuda.tensor.ai

import scuda.tensor. { Tensor, GeneralFunction }
import scuda.tensor.storage.Storage

import scuda.tensor.cpu.ArrayStorage
import scuda.tensor.cuda.CudaStorage
import scuda.tensor.storage.stableSoftmaxGrad
import scuda.tensor.storage.stableSoftmax


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

object ForwardLayer:
	def apply(inFeatures: Int, outFeatures: Int, device: String = "cpu") =
		val rand = scala.util.Random()
		val ub = scala.math.sqrt(6.0 / (inFeatures + outFeatures)).toFloat
		val lb = - ub

		val w = Tensor.fill(Seq(outFeatures, inFeatures), lb + (ub - lb) * rand.nextFloat(), device, true)
		val b = Tensor.fill(Seq(outFeatures, 1), lb + (ub - lb) * rand.nextFloat(), device, true)
		new ForwardLayer(w, b)

class ReLU() extends ReplicatibleFunction:
	def apply(x: Tensor): Tensor = 
		new Tensor(new GeneralFunction {
				lazy val args: Seq[Tensor] = Seq(x)
				lazy val forward = scuda.tensor.storage.relu(x.storage)
				def backward(arg: Tensor, chainGrad: Storage) = 
					scuda.tensor.storage.reluGrad(forward, chainGrad)
			}, x.hasVar)

	def replicate(grad: Map[Tensor, Storage], opt: Optimizer): ReplicatibleFunction = ReLU()

class Sequential(layers: Seq[ReplicatibleFunction]) extends ReplicatibleFunction:
	def apply(x: Tensor): Tensor = 
		var res = x
		layers.foreach{ l =>
			res = l(res) 
		}
		res	
	def replicate(grad: Map[Tensor, Storage], opt: Optimizer) =
		val nSeq = layers.map(l => l.replicate(grad, opt))
		Sequential(nSeq)

class StableSoftmax() extends ReplicatibleFunction:
	def apply(x: Tensor): Tensor = 
		new Tensor(new GeneralFunction {
				lazy val args: Seq[Tensor] = Seq(x)
				lazy val forward = stableSoftmax(x.storage)
				def backward(arg: Tensor, chainGrad: Storage) = stableSoftmaxGrad(forward, chainGrad)
			}, x.hasVar)
	def replicate(grad: Map[Tensor, Storage], opt: Optimizer): ReplicatibleFunction = StableSoftmax()

def heightExpander(x: Tensor, n: Int): Tensor =
	val host = x.storage match
		case _: CudaStorage  => "cuda"
		case _: ArrayStorage => "cpu"
		
	val ones = Tensor(Storage.ones(Seq(n, 1), host), false)

	ones ** (x.T)

	
