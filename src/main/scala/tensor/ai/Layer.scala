package scuda.tensor.ai

import scala.collection.parallel.CollectionConverters._

import scuda.tensor. { Tensor, GeneralFunction }
import scuda.tensor.storage.Storage

import scuda.tensor.cpu.ArrayStorage
import scuda.tensor.cuda.CudaStorage
import scuda.tensor.storage.stableSoftmaxGrad
import scuda.tensor.storage.stableSoftmax
import scuda.tensor.storage.conv2D
import scuda.tensor.storage.conv2dGradW
import scuda.tensor.storage.conv2dGradX


trait ReplicatibleFunction:
	def apply(x: Tensor): Tensor
	def replicate(grad: Map[Tensor, Storage], opt: Optimizer): ReplicatibleFunction

class ForwardLayer(val w: Tensor, val b: Tensor) extends ReplicatibleFunction:
	require(w.storage.shape.length == 2)
	require(b.storage.shape.length == 2)
	require(w.storage.shape(0) == w.storage.shape(0))
	def apply(x: Tensor): Tensor = x ** w.T + b.T

	def replicate(grad: Map[Tensor, Storage], opt: Optimizer): ForwardLayer = 
		val newW = opt(w.storage, grad(w))
		val newB = opt(b.storage, grad(b))
		new ForwardLayer(Tensor(newW, w.hasVar), Tensor(newB, w.hasVar))

object ForwardLayer:
	def apply(inFeatures: Int, outFeatures: Int)(using device: String = "cpu") =
		require(inFeatures > 0, "ForwardLayer: in features must be > 0")
		require(outFeatures > 0, "ForwardLayer: out features must be > 0")
		val rand = scala.util.Random()
		val ub = scala.math.sqrt(6.0 / (inFeatures + outFeatures)).toFloat
		val lb = - ub

		val w = Tensor.fill(Seq(outFeatures, inFeatures), lb + (ub - lb) * rand.nextFloat(), true)
		val b = Tensor.fill(Seq(outFeatures, 1), lb + (ub - lb) * rand.nextFloat(), true)
		new ForwardLayer(w, b)

class Conv2DLayer(val w: Tensor, val b: Tensor, stride: Int = 1, padding: Int = 0) extends ReplicatibleFunction:
	require(w.storage.shape.length == 4)
	require(padding < w.storage.shape.last)
	def apply(x: Tensor): Tensor = 
		require(x.storage.shape.length == 4)
		new Tensor(new GeneralFunction{
		val args: Seq[Tensor] = Seq(w, x)
		val forward: Storage = conv2D(x.storage, w.storage, stride, padding)

		def backward(arg: Tensor, chainGrad: Storage): Storage = 
			require(forward.shape == chainGrad.shape, s"Gradient Conv2D can't be found if forward.shape != chainGrad.shape.\nchainGrad: ${chainGrad.shape}, forward: ${forward.shape}")
			if      w == arg then conv2dGradW(x.storage, chainGrad, w.shape.last, stride, padding)
			else if x == arg then conv2dGradX(w.storage, chainGrad, x.shape.last, stride, padding)
			else                  Storage.zeros(arg.storage)

	}, w.hasVar || x.hasVar) + b
	def replicate(grad: Map[Tensor, Storage], opt: Optimizer): ReplicatibleFunction = 
		val newW = opt(w.storage, grad(w))
		val newB = opt(b.storage, grad(b))
		new Conv2DLayer(Tensor(newW, w.hasVar), Tensor(newB, w.hasVar))

object Conv2DLayer:
	def apply(inChanels: Int, outChannels: Int, kernelSize: Int = 1, stride: Int = 1, padding: Int = 0)(using device: String = "cpu")=
		// ConvCount X inChanels X kernelSize X kernelSize
		val rand = scala.util.Random()
		val ub = scala.math.sqrt(6.0 / (inChanels * outChannels * kernelSize)).toFloat
		val lb = - ub
		val w = Tensor.fill(Seq(outChannels, inChanels, kernelSize, kernelSize), lb + (ub - lb) * rand.nextFloat(), true)
		val b = Tensor.fill(Seq(1, outChannels, 1, 1), lb + (ub - lb) * rand.nextFloat(), true)
		new Conv2DLayer(w, b, stride, padding)

class ReLU() extends ReplicatibleFunction:
	def apply(x: Tensor): Tensor = 
		new Tensor(new GeneralFunction {
				val args: Seq[Tensor] = Seq(x)
				val forward = scuda.tensor.storage.relu(x.storage)
				def backward(arg: Tensor, chainGrad: Storage) = 
					scuda.tensor.storage.reluGrad(forward, chainGrad)
			}, x.hasVar)

	def replicate(grad: Map[Tensor, Storage], opt: Optimizer): ReplicatibleFunction = ReLU()

class Sigmoid() extends ReplicatibleFunction:
  def apply(x: Tensor): Tensor = 
    new Tensor(new GeneralFunction {
        val args: Seq[Tensor] = Seq(x)
        val forward = scuda.tensor.storage.sigmoid(x.storage)
        def backward(arg: Tensor, chainGrad: Storage) = 
          scuda.tensor.storage.sigmoidGrad(forward, chainGrad)
      }, x.hasVar)

  def replicate(grad: Map[Tensor, Storage], opt: Optimizer): ReplicatibleFunction = Sigmoid()

class Tanh() extends ReplicatibleFunction:
  def apply(x: Tensor): Tensor = 
    new Tensor(new GeneralFunction {
        val args: Seq[Tensor] = Seq(x)
        val forward = scuda.tensor.storage.tanh(x.storage)
        def backward(arg: Tensor, chainGrad: Storage) = 
          scuda.tensor.storage.tanhGrad(forward, chainGrad)
      }, x.hasVar)

  def replicate(grad: Map[Tensor, Storage], opt: Optimizer): ReplicatibleFunction = Tanh()

class Sequential(layers: Seq[ReplicatibleFunction]) extends ReplicatibleFunction:
	def apply(x: Tensor): Tensor = 
		var res = x
		layers.foreach{ l =>
			res = l(res) 
		}
		res	
	def replicate(grad: Map[Tensor, Storage], opt: Optimizer) =
		val nSeq = layers.par.map(l => l.replicate(grad, opt)).seq
		Sequential(nSeq)

class StableSoftmax() extends ReplicatibleFunction:
	def apply(x: Tensor): Tensor = 
		new Tensor(new GeneralFunction {
				val args: Seq[Tensor] = Seq(x)
				val forward = stableSoftmax(x.storage)
				def backward(arg: Tensor, chainGrad: Storage) = stableSoftmaxGrad(forward, chainGrad)
			}, x.hasVar)
	def replicate(grad: Map[Tensor, Storage], opt: Optimizer): ReplicatibleFunction = StableSoftmax()

def heightExpander(x: Tensor, n: Int): Tensor =
	implicit val host = x.storage match
		case _: CudaStorage  => "cuda"
		case _: ArrayStorage => "cpu"
		
	val ones = Tensor(Storage.ones(Seq(n, 1)), false)

	ones ** (x.T)

	
