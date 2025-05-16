package scuda.tensor

import scala.math.{Fractional, Numeric}
import scala.reflect.ClassTag

import scuda.tensor.cpu.ArrayStorage
import scuda.tensor.cuda.CudaStorage


import java.lang.instrument.Instrumentation
import scala.annotation.internal.sharable
import scala.annotation.alpha
import scala.compiletime.ops.float

import scala.collection.parallel.CollectionConverters._
import scala.util.Random
import storage.Storage
import scuda.tensor.utils.reverseBroadcasting

// TODO cat, reshape
// hasVar - is variable is in graph
// hasVar + origin.isEmpty => variable wich will have gradient
case class Tensor(val origin: GeneralFunction, val hasVar: Boolean):
	val storage: Storage = origin.forward
	val shape: Seq[Int] = storage.shape

	override def toString(): String = storage.toString()
	
	def +(other: Tensor) = 
		val a = this
		val b = other
		new Tensor(new GeneralFunction {
			val args: Seq[Tensor] = Seq(a, b)
			val forward = a.storage + b.storage
			def backward(arg: Tensor, chainGrad: Storage) =
				if forward.shape != chainGrad.shape then 
					throw new Exception(s"Gradient a + b can't be found if forward.shape != chainGrad.shape.\nchainGrad: ${chainGrad.shape}, forward: ${forward.shape}")
				if      a == arg then reverseBroadcasting(chainGrad, a.storage.shape)
				else if b == arg then reverseBroadcasting(chainGrad, b.storage.shape)
				else                  Storage.zeros(arg.storage)
		}, a.hasVar || b.hasVar)

	def -(other: Tensor) = 
		val a = this
		val b = other
		new Tensor(new GeneralFunction {
			val args: Seq[Tensor] = Seq(a, b)
			val forward = a.storage - b.storage
			def backward(arg: Tensor, chainGrad: Storage) =
				if forward.shape != chainGrad.shape then 
					throw new Exception(s"Gradient a - b can't be found if forward.shape != chainGrad.shape.\nchainGrad: ${chainGrad.shape}, forward: ${forward.shape}")
				if      a == arg then reverseBroadcasting(chainGrad, a.storage.shape)
				else if b == arg then reverseBroadcasting(-chainGrad, b.storage.shape)
				else                  Storage.zeros(arg.storage)
		}, a.hasVar || b.hasVar)

	def *(other: Tensor) = 
		val a = this
		val b = other
		new Tensor(new GeneralFunction {
			val args: Seq[Tensor] = Seq(a, b)
			val forward = a.storage * b.storage
			def backward(arg: Tensor, chainGrad: Storage) =
				if forward.shape != chainGrad.shape then 
					throw new Exception(s"Gradient a * b can't be found if forward.shape != chainGrad.shape.\nchainGrad: ${chainGrad.shape}, forward: ${forward.shape}")
				if      a == arg then reverseBroadcasting(chainGrad * b.storage, a.storage.shape)
				else if b == arg then reverseBroadcasting(chainGrad * a.storage, b.storage.shape)
				else                  Storage.zeros(arg.storage)

		}, a.hasVar || b.hasVar)

	def /(other: Tensor) = 
		val a = this
		val b = other
		new Tensor(new GeneralFunction {
			val args: Seq[Tensor] = Seq(a, b)
			val forward = a.storage / b.storage
			def backward(arg: Tensor, chainGrad: Storage) =
				if forward.shape != chainGrad.shape then 
					throw new Exception(s"Gradient a / b can't be found if forward.shape != chainGrad.shape.\nchainGrad: ${chainGrad.shape}, forward: ${forward.shape}")
				if      a == arg then reverseBroadcasting(chainGrad / b.storage, a.storage.shape)
				else if b == arg then reverseBroadcasting(-chainGrad * a.storage / (b.storage pow 2), b.storage.shape)
				else                  Storage.zeros(arg.storage)
		}, a.hasVar || b.hasVar)

	def **(other: Tensor) = 
		val a = this
		val b = other
		new Tensor(new GeneralFunction {
			val args: Seq[Tensor] = Seq(a, b)
			val forward = a.storage ** b.storage
			def backward(arg: Tensor, chainGrad: Storage) =
				if forward.shape != chainGrad.shape then 
					throw new Exception(s"Gradient a ** b can't be found if forward.shape != chainGrad.shape.\nchainGrad: ${chainGrad.shape}, forward: ${forward.shape}")
				if a == arg then        chainGrad ** (b.storage.T)
				else if b == arg then   (a.storage.T) ** chainGrad
				else                    Storage.zeros(arg.storage)
		}, a.hasVar || b.hasVar)
	
	def +(alpha: Float) =
		val a = this
		new Tensor(new GeneralFunction {
			val args: Seq[Tensor] = Seq(a)
			val forward = storage + alpha
			def backward(arg: Tensor, chainGrad: Storage) = 
				if forward.shape != chainGrad.shape then 
					throw new Exception(s"Gradient a + alpha can't be found if forward.shape != chainGrad.shape.\nchainGrad: ${chainGrad.shape}, forward: ${forward.shape}")
				if a == arg then  chainGrad
				else              Storage.zeros(arg.storage)
		}, hasVar)

	def -(alpha: Float) =
		val a = this
		new Tensor(new GeneralFunction {
			val args: Seq[Tensor] = Seq(a)
			val forward = storage - alpha
			def backward(arg: Tensor, chainGrad: Storage) = 
				if forward.shape != chainGrad.shape then 
					throw new Exception(s"Gradient a - alpha can't be found if forward.shape != chainGrad.shape.\nchainGrad: ${chainGrad.shape}, forward: ${forward.shape}")
				if a == arg then  chainGrad
				else              Storage.zeros(arg.storage)
		}, hasVar)
	
	def *(alpha: Float) =
		val a = this
		new Tensor(new GeneralFunction {
			val args: Seq[Tensor] = Seq(a)
			val forward = storage * alpha
			def backward(arg: Tensor, chainGrad: Storage) = 
				if forward.shape != chainGrad.shape then 
					throw new Exception(s"Gradient a * alpha can't be found if forward.shape != chainGrad.shape.\nchainGrad: ${chainGrad.shape}, forward: ${forward.shape}")
				if a == arg then  chainGrad * alpha
				else              Storage.zeros(arg.storage)
		}, hasVar)
	
	def /(alpha: Float) =
		val a = this
		new Tensor(new GeneralFunction {
			val args: Seq[Tensor] = Seq(a)
			val forward = storage / alpha
			def backward(arg: Tensor, chainGrad: Storage) = 
				if forward.shape != chainGrad.shape then 
					throw new Exception(s"Gradient a / alpha can't be found if forward.shape != chainGrad.shape.\nchainGrad: ${chainGrad.shape}, forward: ${forward.shape}")
				if a == arg then  chainGrad / alpha
				else              Storage.zeros(arg.storage)
		}, hasVar)

	def unary_- =
		val a = this
		new Tensor(new GeneralFunction {
			val args: Seq[Tensor] = Seq(a)
			val forward = -storage
			def backward(arg: Tensor, chainGrad: Storage) =
				if forward.shape != chainGrad.shape then 
					throw new Exception(s"Gradient -a can't be found if forward.shape != chainGrad.shape.\nchainGrad: ${chainGrad.shape}, forward: ${forward.shape}")
				if a == arg then  -chainGrad
				else              Storage.zeros(arg.storage)
		}, hasVar)

	def toCpu = 
		val a = this
		new Tensor(new GeneralFunction {
			val args: Seq[Tensor] = Seq(a)
			val forward = storage.toCpu
			def backward(arg: Tensor, chainGrad: Storage) = 
				if forward.shape != chainGrad.shape then 
					throw new Exception(s"Gradient a.toCpu can't be found if forward.shape != chainGrad.shape.\nchainGrad: ${chainGrad.shape}, forward: ${forward.shape}")
				storage match
					case x: ArrayStorage if x == arg.storage => chainGrad
					case x: CudaStorage if x == arg.storage  => chainGrad.toCuda
					case _                         => Storage.zeros(arg.storage)
		}, hasVar)
	
	def toCuda = 
		val a = this
		new Tensor(new GeneralFunction {
			val args: Seq[Tensor] = Seq(a)
			val forward = storage.toCuda
			def backward(arg: Tensor, chainGrad: Storage) = 
				if forward.shape != chainGrad.shape then 
					throw new Exception(s"Gradient a.toCuda can't be found if forward.shape != chainGrad.shape.\nchainGrad: ${chainGrad.shape}, forward: ${forward.shape}")
				storage match
					case x: ArrayStorage if x == arg.storage => chainGrad.toCpu
					case x: CudaStorage if x == arg.storage  => chainGrad
					case _                         => Storage.zeros(arg.storage)
		}, hasVar)

	def pow(n: Float) = 
		val a = this
		new Tensor(new GeneralFunction {
			val args: Seq[Tensor] = Seq(a)
			val forward = storage pow n
			def backward(arg: Tensor, chainGrad: Storage) = 
				if forward.shape != chainGrad.shape then 
					throw new Exception(s"Gradient a pow n can't be found if forward.shape != chainGrad.shape.\nchainGrad: ${chainGrad.shape}, forward: ${forward.shape}")
				if a == arg then  chainGrad * n * storage.pow(n - 1)
				else              Storage.zeros(arg.storage)
				
		}, hasVar)

	def sum: Tensor = 
		val a = this
		new Tensor(new GeneralFunction {
			val args: Seq[Tensor] = Seq(a)
			val forward = storage.sum
			def backward(arg: Tensor, chainGrad: Storage) = 
				if forward.shape != chainGrad.shape then 
					throw new Exception(s"Gradient a.sum can't be found if forward.shape != chainGrad.shape.\nchainGrad: ${chainGrad.shape}, forward: ${forward.shape}")
				if a == arg then  Storage.fill(a.storage, chainGrad.item)
				else              Storage.zeros(arg.storage)
		}, hasVar)

	def T: Tensor =
		val a = this
		new Tensor(new GeneralFunction {
			val args: Seq[Tensor] = Seq(a)
			val forward = storage.T
			def backward(arg: Tensor, chainGrad: Storage) = 
				if forward.shape != chainGrad.shape then 
					throw new Exception(s"Gradient a.T can't be found if forward.shape != chainGrad.shape.\nchainGrad: ${chainGrad.shape}, forward: ${forward.shape}")
				if a == arg then  chainGrad.T
				else              Storage.zeros(arg.storage)
		}, hasVar)

	def item: Float = storage.item

	def historyDrop(hasVar: Boolean): Tensor =
		new Tensor(new GeneralFunction {
			val args: Seq[Tensor] = Seq()
			val forward = storage
			def backward(arg: Tensor, chainGrad: Storage) = 
				if forward.shape != chainGrad.shape then 
					throw new Exception(s"Gradient a.historyDrop can't be found if forward.shape != chainGrad.shape.\nchainGrad: ${chainGrad.shape}, forward: ${forward.shape}")
				Storage.zeros(arg.storage)
		}, hasVar)

	def flatten(from: Int = 0, to: Int = storage.shape.length) =
		val a = this
		new Tensor(new GeneralFunction {
			val args: Seq[Tensor] = Seq(a)
			val forward = storage.flatten(from, to)
			def backward(arg: Tensor, chainGrad: Storage) = 
				if forward.shape != chainGrad.shape then 
					throw new Exception(s"Gradient a.flatten can't be found if forward.shape != chainGrad.shape.\nchainGrad: ${chainGrad.shape}, forward: ${forward.shape}")
				if a == arg then  chainGrad.reshape(a.storage.shape)
				else              Storage.zeros(arg.storage)
		}, hasVar)

object Tensor:
	def apply(data: Storage, isGrad: Boolean): Tensor = 
		lazy val res: Tensor = new Tensor(new GeneralFunction {
			val args: Seq[Tensor] = Seq()
			val forward: Storage = data
			def backward(arg: Tensor, chainGrad: Storage): Storage = 
				if forward.shape != chainGrad.shape then 
					throw new Exception(s"Gradient x can't be found if forward.shape != chainGrad.shape.\nchainGrad: ${chainGrad.shape}, forward: ${forward.shape}")
				if arg == res && isGrad then  Storage.ones(forward) 
				else                          Storage.zeros(arg.storage)
		}, isGrad)
		res

	def apply(data: Iterable[Float], shape: Seq[Int], isGrad: Boolean = false)(using  device: String = "cpu"): Tensor = 
		apply(Storage(data, shape), isGrad)
	
	def fill(shape: Seq[Int], v: =>Float, isGrad: Boolean = false)(using device: String = "cpu") = 
		apply(Storage.fill(shape, v), isGrad)

	def ones(shape: Seq[Int], isGrad: Boolean = false)(using device: String = "cpu") =
		fill(shape, 1, isGrad)	

	def zeros(shape: Seq[Int], isGrad: Boolean = false)(using device: String = "cpu") =
		fill(shape, 0, isGrad)

	def rand(shape: Seq[Int], isGrad: Boolean = false)(using device: String = "cpu") =
		val r = Random()
		fill(shape, r.nextFloat(), isGrad)
	def arange(n: Int, isGrad: Boolean = false) = 
		apply(Storage((0 until n).map(_.toFloat), Seq(n)), isGrad)
		


		










