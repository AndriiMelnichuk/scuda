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

	
	// def /(other: Tensor) = new Tensor(
	// 	// () => 
	// 	GeneralFunction(
	// 		Seq(this, other),
	// 		this.storage / other.storage,
	// 		Seq(
	// 			() => 1f / other.storage,
	// 			() => -1f * this.storage / (other.storage * other.storage)
	// 		)
	// 	)
	// )
	
	// def T: Tensor = ???

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
	
	// def +(alpha: Float) = new Tensor(() => this.storage + alpha)
	// def -(alpha: Float) = new Tensor(() => this.storage - alpha)
	// def *(alpha: Float) = new Tensor(() => this.storage * alpha)
	// def /(alpha: Float) = new Tensor(() => this.storage / alpha)
	
	// def toCpu(): Tensor = new Tensor(() => this.storage.toCpu())
	// def toCuda(): Tensor = new Tensor(() => this.storage.toCuda())

	def sum: Tensor = 
		val a = this
		new Tensor(new GeneralFunction {
			lazy val args: Seq[Tensor] = Seq(a)
			lazy val forward = args(0).storage.sum
			def backward(arg: Tensor, chainGrad: Storage) = 
				chainGrad match
					case x: ArrayStorage => Storage.fill(a.storage.shape, x.storage(0))
					case _: CudaStorage => ???
					// TODO CUDA STORAGE
		}, hasVar)

	def T: Tensor =
		val a = this
		new Tensor(new GeneralFunction {
			lazy val args: Seq[Tensor] = Seq(a)
			lazy val forward = a.storage.T
			def backward(arg: Tensor, chainGrad: Storage) = chainGrad.T
		}, hasVar)

	def item: Float = storage.item

object Tensor:
//     def apply(data: Seq[Float]) = new Tensor(() => ArrayStorage(data.toArray, Seq(data.length)))
	def apply(data: Seq[Float], shape: Seq[Int], device: String = "cpu", isGrad: Boolean = false) = 
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

//     def apply(data: Seq[Float], shape: Seq[Int]) = new Tensor(() => {
//         if shape.product != data.length then throw new Exception("Elligal shape")
//         if shape.isEmpty then throw new Exception("Shape empty")
//         ArrayStorage(data.toArray, shape)
//     })

//     def apply(data: Seq[Float], storageType: String) = new Tensor(() => {
//         storageType.toLowerCase match
//             case "cpu" => ArrayStorage(data.toArray, Seq(data.length))
//             case "cuda" => CudaStorage(data.toArray, Seq(data.length))
//             case _ => throw new Exception("Unknown device type")
//     })

//     def apply(data: Seq[Float], shape: Seq[Int], storageType: String) = new Tensor(() => {
//         if shape.product != data.length then throw new Exception("Elligal shape")
//         if shape.isEmpty then throw new Exception("Shape empty")
//         storageType.toLowerCase match
//             case "cpu" => ArrayStorage(data.toArray, shape)
//             case "cuda" => CudaStorage(data.toArray, shape)
//             case _ => throw new Exception("Unknown device type")
//     })
	
//     def fill(shape: Seq[Int])(v: Float) = new Tensor(() => {
//         if shape.isEmpty then throw new Exception("Shape empty")
//         if shape.filter(_ <= 0).nonEmpty then Exception("Elligal shape")
//         ArrayStorage(Array.fill(shape.product)(v), shape)
//         ???
//     })

		


		










