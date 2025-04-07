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

// hasVar - is variable is in graph
// hasVar + origin.isEmpty => variable wich will have gradient
case class Tensor(val origin: GeneralFunction, val hasVar: Boolean):
	lazy val storage: Storage = origin.forward
	
	override def toString(): String = storage.toString()
	
	def +(other: Tensor) = 
		val a = this
		val b = other
		new Tensor(new GeneralFunction {
			lazy val args: Seq[Tensor] = Seq(a, b)
			lazy val forward = args(0).storage + args(1).storage
			def backward(arg: Tensor, chainGrad: Storage) = chainGrad
		}, a.hasVar || b.hasVar)

	def -(other: Tensor) = 
		val a = this
		val b = other
		new Tensor(new GeneralFunction {
			lazy val args: Seq[Tensor] = Seq(a, b)
			lazy val forward = args(0).storage - args(1).storage
			def backward(arg: Tensor, chainGrad: Storage) =
				if arg == a then
					chainGrad
				else
					chainGrad * -1
		}, a.hasVar || b.hasVar)

	def *(other: Tensor) = 
		val a = this
		val b = other
		new Tensor(new GeneralFunction {
			lazy val args: Seq[Tensor] = Seq(a, b)
			lazy val forward = args(0).storage * args(1).storage
			def backward(arg: Tensor, chainGrad: Storage) =
				if arg == a then
					chainGrad * b.storage
				else
					chainGrad * a.storage
		}, a.hasVar || b.hasVar)

	def /(other: Tensor) = 
		val a = this
		val b = other
		new Tensor(new GeneralFunction {
			lazy val args: Seq[Tensor] = Seq(a, b)
			lazy val forward = args(0).storage / args(1).storage
			def backward(arg: Tensor, chainGrad: Storage) =
				if arg == a then
					chainGrad / b.storage
				else
					-chainGrad * a.storage / (b.storage * b.storage)
		}, a.hasVar || b.hasVar)

	def **(other: Tensor) = 
		val a = this
		val b = other
		new Tensor(new GeneralFunction {
			lazy val args: Seq[Tensor] = Seq(a, b)
			lazy val forward = args(0).storage ** args(1).storage
			def backward(arg: Tensor, chainGrad: Storage) =
				if a == arg then 
					chainGrad ** (b.storage.T)
				else
					(a.storage.T) ** chainGrad
		}, a.hasVar || b.hasVar)
	
	def +(alpha: Float) =
		val a = this
		new Tensor(new GeneralFunction {
			lazy val args: Seq[Tensor] = Seq(a)
			lazy val forward = storage + alpha
			def backward(arg: Tensor, chainGrad: Storage) = chainGrad
		}, hasVar)

	def -(alpha: Float) =
		val a = this
		new Tensor(new GeneralFunction {
			lazy val args: Seq[Tensor] = Seq(a)
			lazy val forward = storage - alpha
			def backward(arg: Tensor, chainGrad: Storage) = chainGrad
		}, hasVar)
	
	def *(alpha: Float) =
		val a = this
		new Tensor(new GeneralFunction {
			lazy val args: Seq[Tensor] = Seq(a)
			lazy val forward = storage * alpha
			def backward(arg: Tensor, chainGrad: Storage) = 
				Storage.fill(chainGrad, alpha) * chainGrad
		}, hasVar)
	
	def /(alpha: Float) =
		val a = this
		new Tensor(new GeneralFunction {
			lazy val args: Seq[Tensor] = Seq(a)
			lazy val forward = storage * alpha
			def backward(arg: Tensor, chainGrad: Storage) = 
				Storage.fill(chainGrad, 1/alpha) * chainGrad
		}, hasVar)

	def unary_- =
		val a = this
		new Tensor(new GeneralFunction {
			lazy val args: Seq[Tensor] = Seq(a)
			lazy val forward = -storage
			def backward(arg: Tensor, chainGrad: Storage) = -chainGrad
		}, hasVar)

	// TODO	Must be tested
	def toCpu = 
		val a = this
		new Tensor(new GeneralFunction {
			lazy val args: Seq[Tensor] = Seq(a)
			lazy val forward = storage.toCpu()
			def backward(arg: Tensor, chainGrad: Storage) = 
				storage match
					case _: ArrayStorage => chainGrad
					case _: CudaStorage => chainGrad.toCuda()
		}, hasVar)
	
	def toCuda = 
		val a = this
		new Tensor(new GeneralFunction {
			lazy val args: Seq[Tensor] = Seq(a)
			lazy val forward = storage.toCuda()
			def backward(arg: Tensor, chainGrad: Storage) = 
				storage match
					case _: ArrayStorage => chainGrad.toCpu()
					case _: CudaStorage => chainGrad
		}, hasVar)

	def sum: Tensor = 
		val a = this
		new Tensor(new GeneralFunction {
			lazy val args: Seq[Tensor] = Seq(a)
			lazy val forward = args(0).storage.sum
			def backward(arg: Tensor, chainGrad: Storage) = 
				Storage.fill(a.storage, chainGrad.item)
		}, hasVar)

	def T: Tensor =
		val a = this
		new Tensor(new GeneralFunction {
			lazy val args: Seq[Tensor] = Seq(a)
			lazy val forward = a.storage.T
			def backward(arg: Tensor, chainGrad: Storage) = chainGrad.T
		}, hasVar)

	def item: Float = storage.item

	def historyDrop(hasVar: Boolean): Tensor =
		new Tensor(new GeneralFunction {
			lazy val args: Seq[Tensor] = Seq()
			lazy val forward = storage
			def backward(arg: Tensor, chainGrad: Storage) = 
				Storage.ones(storage)
		}, hasVar)

object Tensor:
	def apply(data: Iterable[Float], shape: Seq[Int], device: String = "cpu", isGrad: Boolean = false) = 
		new Tensor(new GeneralFunction {
			lazy val args: Seq[Tensor] = Seq()
			lazy val forward: Storage = Storage(data, shape, device)
			def backward(argument: Tensor, chainGrad: Storage): Storage = 
				argument.storage match
					case forward => Storage.ones(forward)
		}, isGrad)
	
	def apply(data: Storage, isGrad: Boolean) = 
		new Tensor(new GeneralFunction {
			lazy val args: Seq[Tensor] = Seq()
			lazy val forward: Storage = data
			def backward(argument: Tensor, chainGrad: Storage): Storage = 
				argument.storage match
					case forward => Storage.ones(forward)
		}, isGrad)

	def fill(shape: Seq[Int], v: =>Float, device: String = "cpu", isGrad: Boolean = false) = 
		new Tensor(new GeneralFunction {
			lazy val args: Seq[Tensor] = Seq()
			lazy val forward: Storage = Storage.fill(shape, v, device)
			def backward(argument: Tensor, chainGrad: Storage): Storage = 
				argument.storage match
					case forward => Storage.ones(forward)
		}, false)

	def ones(shape: Seq[Int], device: String = "cpu", isGrad: Boolean = false) =
		fill(shape, 1, device, isGrad)	

	def zeros(shape: Seq[Int], device: String = "cpu", isGrad: Boolean = false) =
		fill(shape, 0, device, isGrad)

	def rand(shape: Seq[Int], device: String = "cpu", isGrad: Boolean = false) =
		val r = Random()
		fill(shape, r.nextFloat(), device, isGrad)

		


		










