package scuda.tensor.storage

import scuda.tensor.cuda.*
import scuda.tensor.cpu.*


def relu(x: Storage): Storage = x match
	case x: CudaStorage   => scuda.tensor.cuda.relu(x)
	case x: ArrayStorage  => scuda.tensor.cpu.relu(x)
	case _                => throw new IllegalArgumentException("Unsupported storage type for ReLU")

def reluGrad(pr: Storage, cg: Storage): Storage = (pr, cg) match
	case (x: ArrayStorage, y: ArrayStorage) => scuda.tensor.cpu.reluGrad(x, y)
	case (x: CudaStorage, y: CudaStorage)   => scuda.tensor.cuda.reluGrad(x, y)
	case _                                  => throw new IllegalArgumentException("Mismatched or unsupported storage types for reluGrad")

def sigmoid(x: Storage): Storage = x match
	case x: ArrayStorage => scuda.tensor.cpu.sigmoid(x)
	case x: CudaStorage  => scuda.tensor.cuda.sigmoid(x)
	case _               => throw new IllegalArgumentException("Mismatched or unsupported storage types for sigmoid")

def sigmoidGrad(sigmoid: Storage, chainGrad: Storage): Storage =
  (sigmoid - (sigmoid pow 2f)) * chainGrad

def tanh(x: Storage): Storage = x match
	case x: ArrayStorage => scuda.tensor.cpu.tanh(x)
	case x: CudaStorage  => scuda.tensor.cuda.tanh(x)
	case _               => throw new IllegalArgumentException("Mismatched or unsupported storage types for tanh")

def tanhGrad(tanh: Storage, chainGrad: Storage): Storage =
  (-(tanh pow 2f) + 1f) * chainGrad

def crossEntropyLoss(pr: Storage, target: Storage): Storage = (pr, target) match
	case (x: ArrayStorage, y: ArrayStorage) => scuda.tensor.cpu.crossEntropyLoss(x, y)
	case (x: CudaStorage, y: CudaStorage)   => scuda.tensor.cuda.crossEntropyLoss(x, y)
	case _                                  => throw new IllegalArgumentException("Mismatched or unsupported storage types for crossEntropyLoss")
	
def crossEntropyLossGrad(pr: Storage, target: Storage, chainGrad: Storage): Storage = (pr, target, chainGrad) match
	case (x: ArrayStorage, y: ArrayStorage, z: ArrayStorage) => scuda.tensor.cpu.crossEntropyLossGrad(x, y, z)
	case (x: CudaStorage, y: CudaStorage, z: CudaStorage)    => scuda.tensor.cuda.crossEntropyLossGrad(x, y, z)
	case _                                  => throw new IllegalArgumentException("Mismatched or unsupported storage types for crossEntropyLossGrad")

def stableSoftmax(x: Storage): Storage = x match
	case x: ArrayStorage => scuda.tensor.cpu.stableSoftmax(x) 
	case x: CudaStorage  => scuda.tensor.cuda.stableSoftmax(x)
	case _               => throw new IllegalArgumentException("Mismatched or unsupported storage types for stableSoftmax")

def stableSoftmaxGrad(sm: Storage, cg: Storage): Storage = (sm, cg) match
	case (x: ArrayStorage, y: ArrayStorage) => scuda.tensor.cpu.stableSoftmaxGrad(x, y)
	case (x: CudaStorage, y: CudaStorage)   => scuda.tensor.cuda.stableSoftmaxGrad(x, y)
	case _                                  => throw new IllegalArgumentException("Mismatched or unsupported storage types for crossEntropyLossGrad")

def conv2D(x: Storage, w: Storage, b: Storage, kernelSize: Int = 1, stride: Int = 1, padding: Int = 0): Storage = 
	def createFeatureForConv(x: Storage, lhiX: Int, lhiY: Int, imgN: Int, kernelSize: Int): Seq[Float] = 
		x match
			case x: CudaStorage  => scuda.tensor.cuda.createFeatureForConv(x, lhiX, lhiY, imgN, kernelSize)
			case x : ArrayStorage => scuda.tensor.cpu.createFeatureForConv(x, lhiX, lhiY, imgN, kernelSize)
	def convPerFMCount(w: Int): Int =
		(w + 2 * padding - kernelSize) / stride + 1
	def indexForConvSelector(w: Int): Seq[Int] =
		(0 until convPerFMCount(w)).map( _ * stride - padding)
	def createFullFeatureTensor: Storage =
		val h = x.shape(2)
		val w = x.shape(3)

		val res = for 
			z <- 0 until x.shape(0)
			y <- indexForConvSelector(h)
			xx <- indexForConvSelector(w)
		yield
			createFeatureForConv(x, xx, y, z, kernelSize)

		val m = res.length
		val n = res(0).length
		val t = res.flatten
		x match
			case _: CudaStorage  => CudaStorage(res.flatten, Seq(m,n))
			case _: ArrayStorage => ArrayStorage(res.flatten, Seq(m,n))

	val prX    = createFullFeatureTensor
	val imC    = x.shape(0)
	val width  = x.shape(3)
	val height = x.shape(2)
	val fo     = w.shape(0)
	val prW    = w.flatten(1).T
	
	(prX ** prW + b).T
	.reshape(fo, imC, convPerFMCount(height) * convPerFMCount(width))
	.split(1)
	.reduce(_.cat(_, 0))
	.reshape(imC,fo,convPerFMCount(height), convPerFMCount(width))