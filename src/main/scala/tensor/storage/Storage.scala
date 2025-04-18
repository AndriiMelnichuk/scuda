package scuda.tensor.storage

import scuda.tensor.cpu.ArrayStorage
import scuda.tensor.cuda.CudaStorage

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

	def T: Storage
	def toCpu(): ArrayStorage
	def toCuda(): CudaStorage

	def sum: Storage

	def item: Float

	def unary_- : Storage

	def pow(n: Float): Storage
	
	def cat(st: Storage, dim: Int = 0): Storage

object Storage:
	def apply(data: Iterable[Float], shape: Seq[Int], device: String = "cpu") = 
		device match 
			case "cpu" => ArrayStorage(data, shape)
			case "cuda" => CudaStorage(data, shape)
			case _ => throw new Exception("Uncnown device")

	def fill(shape: Seq[Int], value: =>Float, dhost: String = "cpu"): Storage =
		dhost match
			case "cpu" => ArrayStorage.fill(shape, value)
			case "cuda" => CudaStorage.fill(shape, value)
			case _ => throw new Exception("Unknown device host")

	def ones(shape: Seq[Int], dhost: String = "cpu"): Storage =
		fill(shape, 1, dhost)
	
	def zeros(shape: Seq[Int], dhost: String = "cpu"): Storage =
		fill(shape, 0, dhost)

	def fill(origin: Storage, value: Float): Storage =
		origin match
			case _: ArrayStorage => fill(origin.shape, value)
			case _: CudaStorage => fill(origin.shape, value, "cuda")

	def ones(origin: Storage) =
		fill(origin, 1)

	def zeros(origin: Storage) = 
		fill(origin, 0)
	
	def rand(shape: Seq[Int], dhost: String = "cpu"): Storage =
		dhost match
			case "cpu" => ArrayStorage.rand(shape)
			case "cuda" => CudaStorage.rand(shape)
			case _ => throw new Exception("Unknown device host")
