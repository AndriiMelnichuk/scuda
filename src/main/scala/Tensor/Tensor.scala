package scuda.Tensor

import scala.math.{Fractional, Numeric}
import scala.reflect.ClassTag



import java.lang.instrument.Instrumentation
import scala.annotation.internal.sharable
import scala.annotation.alpha
import scala.compiletime.ops.float

import scala.collection.parallel.CollectionConverters._

class Tensor(val origin: GeneralFunction, val isGrad: Boolean):
	lazy val storage: Storage = origin.forward
	
	override def toString(): String = storage.toString()
	
	def +(other: Tensor) = 
		val bf = () => Storage.ones(this.storage)
		val rf = (last: Storage) => last
		new Tensor(
			GeneralFunction(
				Seq(this, other),
				this.storage + other.storage,
				Seq(bf, bf),
				Seq(rf, rf)
			),
			false
		)

	// def -(other: Tensor) = new Tensor(
	// 	GeneralFunction(
	// 		Seq(this, other),
	// 		this.storage - other.storage,
	// 		Seq(() => onesLike(this), () => onesLike(other) * -1)
	// 	)
	// )

	// def *(other: Tensor) = new Tensor(
	// 	GeneralFunction(
	// 		Seq(this, other),
	// 		this.storage * other.storage,
	// 		Seq(() => other.storage, () => this.storage)
	// 	)
	// )

	
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
	
	def T: Tensor = ???

	def **(other: Tensor) = new Tensor(
		GeneralFunction(
			Seq(this, other),
			this.storage ** other.storage,
			Seq(() => other.storage.T, () => this.storage.T),
			Seq(_ ** other.storage.T, this.storage.T ** _)
		),
		false
	)

		// new Tensor(() => this.storage ** other.storage)
	
	// def +(alpha: Float) = new Tensor(() => this.storage + alpha)
	// def -(alpha: Float) = new Tensor(() => this.storage - alpha)
	// def *(alpha: Float) = new Tensor(() => this.storage * alpha)
	// def /(alpha: Float) = new Tensor(() => this.storage / alpha)
	
	// def toCpu(): Tensor = new Tensor(() => this.storage.toCpu())
	// def toCuda(): Tensor = new Tensor(() => this.storage.toCuda())

	

object Tensor:
//     def apply(data: Seq[Float]) = new Tensor(() => ArrayStorage(data.toArray, Seq(data.length)))
	def apply(data: Seq[Float], shape: Seq[Int], device: String = "cpu", isGrad: Boolean = false) = new Tensor(
		GeneralFunction(
			Seq(),
			Storage(data, shape, device),
			Seq(),
			Seq()
		),
		isGrad
	)

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

		


		










