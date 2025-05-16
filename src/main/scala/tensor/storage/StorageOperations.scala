package scuda.tensor.storage

import scuda.tensor.cuda.*
import scuda.tensor.cpu.*

import scala.collection.parallel.CollectionConverters._


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

def conv2D(x: Storage, w: Storage, stride: Int = 1, padding: Int = 0): Storage = 
	require(stride > 0)
	require(padding > -1)
	def createFeatureForConv(st: Storage, lhiX: Int, lhiY: Int, imgN: Int, kernelSize: Int): Seq[Float] = 
		st match
			case cst: CudaStorage  => scuda.tensor.cuda.createFeatureForConv(cst, lhiX, lhiY, imgN, kernelSize)
			case ast: ArrayStorage => scuda.tensor.cpu.createFeatureForConv(ast, lhiX, lhiY, imgN, kernelSize)
	def convPerFMCount(ww: Int): Int =
		(ww + 2 * padding - w.shape.last) / stride + 1
	def indexForConvSelector(w: Int): Seq[Int] =
		(0 until convPerFMCount(w)).map( _ * stride - padding)

	val imC    = x.shape(0)
	val width  = x.shape(3)
	val height = x.shape(2)
	val fo     = w.shape(0)
	val prW    = w.flatten(1).T

	val res = for 
		z <- 0 until imC
		y <- indexForConvSelector(height)
		xx <- indexForConvSelector(width)
	yield
		createFeatureForConv(x, xx, y, z, w.shape.last)

	val m = res.length
	val n = res(0).length
	val t = res.flatten
	val prX = x match
		case _: CudaStorage  => CudaStorage(res.flatten, Seq(m,n))
		case _: ArrayStorage => ArrayStorage(res.flatten, Seq(m,n))

	(prX ** prW).T
	.reshape(fo, imC, convPerFMCount(height) * convPerFMCount(width))
	.split(1)
	.reduce(_.cat(_, 0))
	.reshape(imC,fo,convPerFMCount(height), convPerFMCount(width))

def conv2dGradW(x: Storage, e: Storage, wWidth: Int, stride: Int = 1, padding: Int = 0): Storage = 
	def expandHeight(x: Storage, layerCount: Int, layerN: Int): Storage = 
		require(x.shape.length == 2)
		implicit val device = x match
			case _: ArrayStorage => "cpu"
			case _: CudaStorage => "cuda"
		
		val len = x.shape.product
		val h = x.shape(0)
		val w = x.shape(1)
		
		var res = x.flatten()
		
		if layerN != 0 then
			val left = Storage.zeros(Seq(w * h * layerN))
			res = left.cat(res)
		if layerN != layerCount - 1 then
			val right = Storage.zeros(Seq(w * h * (layerCount - layerN - 1)))
			res = res.cat(right)
		res = res.reshape(layerCount, h, w)
		res
			
	def expandByZeros(x: Storage, expCount: Int = 0): Storage = 
		if expCount == 0 then x
		else
			implicit val device = x match
				case _: ArrayStorage => "cpu"
				case _: CudaStorage => "cuda"
			val data   = x.toCpu.storage
			val wight  = x.shape.last
			val hight  = x.shape.head
			val nWight = expCount * (wight - 1) + wight
			val nHight = expCount * (hight - 1) + hight

			val expWData = data.grouped(wight)
			.map( l =>
				l.init.flatMap(x => x :: List.fill[Float](expCount)(0)) :+ l.last
			).toList
			
			val rData =
				(expWData.init.flatMap { x =>
					x.toList :: List.fill(expCount)(List.fill(nWight)(0f))
				} :+ expWData.last.toList).flatten

			Storage(rData, Seq(nHight, nWight))
			
	// x - imgCount  X inChanels X ? X ?
	// w - ConvCount X inChanels X ? X ?
	// e - imgCount  X ConvCount X ? X ?
	// 
	val imgCount = x.shape(0)
	val inChanels = x.shape(1)
	val convCount = e.shape(1)

	val res = 
	for 
		imgN <- 0 until imgCount 
		convN <- 0 until convCount
	yield
		val nX = x(Seq(imgN)).unsqueeze()
		val sE = expandByZeros(e(Seq(imgN, convN)), stride - 1)
		(0 until inChanels)
		.map( i => expandHeight(sE, inChanels, i).unsqueeze() )
		.map( nE => conv2D(nX, nE, 1, padding) )
		.reduce(_.cat(_, 1))

	val res2 = res.grouped(convCount).toArray
	.map(_.par.reduce(_.cat(_)))
	.reduce(_ + _)

	if res2.shape.last == wWidth then
		res2
	else
		res2.apply(Seq(-1, -1, 0 until wWidth, 0 until wWidth))
	
def conv2dGradX(w: Storage, e: Storage, xWidth: Int, stride: Int = 1, padding: Int = 0): Storage = 
	def reverse(x: Storage): Storage = 
		implicit val device = x match
			case _: ArrayStorage => "cpu"
			case _: CudaStorage => "cuda"
		Storage(x.toCpu.storage.reverse, x.shape.reverse)
			
	def expandByZeros(x: Storage, expCount: Int = 0, necessaryWidth: Int): Storage = 
		if expCount == 0 then x
		else
			implicit val device = x match
				case _: ArrayStorage => "cpu"
				case _: CudaStorage => "cuda"
			val data   = x.toCpu.storage
			val wight  = x.shape.last
			val hight  = x.shape.head
			val nWight = expCount * (wight - 1) + wight
			val nHight = expCount * (hight - 1) + hight

			val expWData = data.grouped(wight)
			.map( l =>
				(l.init.flatMap(x => x :: List.fill[Float](expCount)(0)) :+ l.last) ++ List.fill[Float](necessaryWidth - nWight - 2)(0)
			).toList
			
			val rData =
				(expWData.init.flatMap { x =>
					x.toList :: List.fill(expCount)(List.fill(necessaryWidth - 2)(0f))
				} :+ expWData.last.toList) 
				:+ (List.fill(necessaryWidth - nWight - 2)(List.fill(necessaryWidth - 2)(0f)).flatten)
			
			Storage(rData.flatten, Seq(necessaryWidth - 2, necessaryWidth - 2))
			
	// w - ConvCount X inChanels X ? X ?
	// e - imgCount  X ConvCount X ? X ?
	val imgCount = e.shape(0)
	val inChanels = w.shape(1)
	val convCount = e.shape(1)
	
	// TODO can be optimize if cat inChanels and then conv2d
	val res = (0 until imgCount).flatMap(imgN => (0 until inChanels).map((imgN, _)))
	.par.map{ (imgN, inChanelsN) =>
		val u = (0 until convCount).par
			.map(i => reverse(w(Seq(i, inChanelsN))).unsqueeze())
			.reduce(_.cat(_))
			.unsqueeze()
		val ne = e(Seq(imgN))
			.split(0).par
			.map(expandByZeros(_, stride - 1, xWidth + 2 * padding).unsqueeze()).reduce(_.cat(_))
			.unsqueeze()
		conv2D(ne, u, 1, u.shape.last - padding - 1)
	}.seq
		
	val res2 = res.grouped(inChanels).toArray.par
		.map(_.par.reduce(_.cat(_,1)))
		.reduce(_.cat(_))

	if res2.shape.last == xWidth then
		res2
	else
		res2(Seq(-1,-1, 0 until xWidth, 0 until xWidth))
	
def broadcasting(x: Storage, shape: Seq[Int]): Storage =
	x match
		case x: ArrayStorage => scuda.tensor.cpu.broadcasting(x, shape)
		case x: CudaStorage => scuda.tensor.cuda.broadcasting(x, shape)
	