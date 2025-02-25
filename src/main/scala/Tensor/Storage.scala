package scuda.Tensor


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

object Storage:
	def apply(data: Seq[Float], shape: Seq[Int], device: String = "cpu") = 
		device match 
			case "cpu" => new ArrayStorage(data.toArray, shape)
			case "cuda" => CudaStorage(data, shape)
			case _ => throw new Exception("Uncnown device")

	def fill(shape: Seq[Int], value: Float, dhost: String = "cpu"): Storage =
		dhost match
			case "cpu" => ArrayStorage(Array.fill(shape.product)(value), shape)
			case "cuda" => CudaStorage(Seq.fill(shape.product)(value), shape)
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
