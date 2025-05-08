package scuda.tensor.storage

import scuda.tensor.cpu.ArrayStorage
import scuda.tensor.cuda.CudaStorage

import scala.collection.parallel.CollectionConverters._

trait Storage:
	val shape: Seq[Int]
	override def toString(): String
	def +(other: Storage): Storage
	def -(other: Storage): Storage
	def *(other: Storage): Storage
	def /(other: Storage): Storage
	def **(other: Storage): Storage

	def +(alpha: Float): Storage
	def -(alpha: Float): Storage
	def *(alpha: Float): Storage
	def /(alpha: Float): Storage
	def pow(n: Float): Storage
	def unary_- : Storage

	// device change
	def toCpu: ArrayStorage
	def toCuda: CudaStorage

	// reduce
	def sum: Storage
	def item: Float
	def split(dim: Int = 0, size: Int = 1): Seq[Storage] = 
		require(dim >= 0)
		require(dim < shape.length)

		val nShape = shape.updated(dim, 1)
		(0 until shape(dim))
		.map{ i =>
			this(Seq.fill(dim)(-1).appended(i))
		}
		.grouped(size).toSeq.par
		.map{ x =>
			x.reduce((x, y) => x.reshape(nShape).cat(y.reshape(nShape), dim) )
		}.seq

	def sum(axis: Int = 0): Storage = 
		split(axis).reduce(_ + _)
		

	def T: Storage
	def reshape(seq: Int*): Storage = 
		reshape(seq.toIterable)
	def reshape(seq: Iterable[Int]): Storage
	def apply(args: Seq[Int | Iterable[Int]]): Storage
	def flatten(from: Int = 0, to: Int = shape.length) =
		require(from >= 0)
		require(from < to)
		val prod = shape.slice(from, to + 1).product
		val nshape = shape.take(from) ++ Seq(prod) ++ shape.drop(to)
		reshape(nshape)

	
	def cat(st: Storage, dim: Int = 0): Storage

object Storage:
	def apply(data: Iterable[Float], shape: Seq[Int])(using device: String = "cpu") = 
		device match 
			case "cpu" => ArrayStorage(data, shape)
			case "cuda" => CudaStorage(data, shape)
			case _ => throw new Exception("Uncnown device")

	def fill(shape: Seq[Int], value: =>Float)(using device: String = "cpu"): Storage =
		device match
			case "cpu" => ArrayStorage.fill(shape, value)
			case "cuda" => CudaStorage.fill(shape, value)
			case _ => throw new Exception("Unknown device host")

	def ones(shape: Seq[Int])(using device: String = "cpu"): Storage =
		fill(shape, 1)
	
	def zeros(shape: Seq[Int])(using device: String = "cpu"): Storage =
		fill(shape, 0)

	def fill(origin: Storage, value: Float): Storage =
		origin match
			case _: ArrayStorage => fill(origin.shape, value)
			case _: CudaStorage => fill(origin.shape, value)(using "cuda")

	def ones(origin: Storage) =
		fill(origin, 1)

	def zeros(origin: Storage) = 
		fill(origin, 0)
	
	def rand(shape: Seq[Int])(using device: String = "cpu"): Storage =
		device match
			case "cpu" => ArrayStorage.rand(shape)
			case "cuda" => CudaStorage.rand(shape)
			case _ => throw new Exception("Unknown device host")

	def arange(n: Int)(using device: String = "cpu"): Storage =
		Storage((0 until n).map(_.toFloat), Seq(n))