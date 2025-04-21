package scuda.tensor.cuda

import scuda.tensor.storage.Storage
import scuda.tensor.cpu.ArrayStorage
import scuda.tensor.utils.{ beautifulArrayprint, device2host, host2device }
import scala.collection.parallel.CollectionConverters._

import jcuda.Pointer
import jcuda.runtime._
import jcuda.runtime.JCuda._
import jcuda.Sizeof
import jcuda.jcublas.*
import scala.collection.View.FlatMap
import scala.util.Random

class CudaStorage(
	val storage: Pointer, 
	val shape: Seq[Int], 
) extends Storage:
	
	override def toString(): String = beautifulArrayprint(device2host(storage, shape), shape)

	override def finalize() = cudaFree(storage)

	private val elementwiseCernelPath = "src/main/resources/tesorElementwiseOperatins.ptx"
	
	def elementwiseOperation(other: Storage, ptxFile: String, opName: String): Storage = 
		if this.shape != other.shape then throw new Exception("Operation cannot be performed if shape of Tensors isn't equal.")
		other match
				case other: CudaStorage => elementwiseCernelExecuter(this, other, ptxFile, opName)
				case _ => throw new Exception("Operation cannot be performed if devise isn't same.")

	def elementwiseScalarOperation(other: Float, ptxFile: String, opName: String): Storage = 
		elementwiseScalarCernelExecuter(this, other, ptxFile, opName)

	def +(other: Storage) =
		elementwiseOperation(other, elementwiseCernelPath, "tensorAddition")
	
	def -(other: Storage): Storage = 
		elementwiseOperation(other, elementwiseCernelPath, "tensorSubtraction")
	
	def *(other: Storage): Storage = 
		elementwiseOperation(other, elementwiseCernelPath, "tensorMultiplication")

	def /(other: Storage): Storage = 
		elementwiseOperation(other, elementwiseCernelPath, "tensorDivision")

	def +(alpha: Float): Storage = 
		elementwiseScalarOperation(alpha, elementwiseCernelPath, "tensorSAddition")
	def -(alpha: Float): Storage = 
		elementwiseScalarOperation(alpha, elementwiseCernelPath, "tensorSSubtraction")

	def *(alpha: Float): Storage = 
		elementwiseScalarOperation(alpha, elementwiseCernelPath, "tensorSMultiplication")


	def /(alpha: Float): Storage = 
		elementwiseScalarOperation(alpha, elementwiseCernelPath, "tensorSDivision")
			
	def **(other: Storage): Storage = 
		if this.shape.length != 2 || other.shape.length != 2 then 
			throw new Exception("Operation cannot be performed if shape of Tensors isn't equal 2.")
		if this.shape(1) != other.shape(0) then 
			throw new Exception("Operation cannot be performed if Tensor A.shape(1) != Tensor B.shape(0)")
		
		other match
			case other: CudaStorage => 
				val m = this.shape(0)
				val k = this.shape(1)
				val n = other.shape(1)
				val res = Pointer()
				cudaMalloc(res, m * n * Sizeof.FLOAT)
				
				val handle = new cublasHandle()
				JCublas2.cublasCreate(handle)
				JCublas2.cublasSgemm(handle, 
					cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N,
					n, m, k,
					Pointer.to(Array[Float](1.0f)),
					other.storage, n,
					this.storage, k,
					Pointer.to(Array[Float](0.0f)),
					res, n)
				new CudaStorage(res, Seq(m, n))
			case _ => 
				throw new Exception("Operation cannot be performed if devise isn't same.")


	def toCpu(): ArrayStorage = new ArrayStorage(device2host(storage, shape), shape)

	def toCuda(): CudaStorage = 
		val res = Pointer()
		cudaMalloc(res, shape.product * Sizeof.FLOAT)
		cudaMemcpy(res, storage, shape.product * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToDevice)
		new CudaStorage(res, shape)

	def sum: CudaStorage = 
		// TODO: realization with CUDA power
		this.toCpu().sum.toCuda()
		
	def T: Storage = 
		if shape.length != 2 then throw new Exception("not 2d Tensor cant be transponed")
		val mBlockSize = 1024
		val m = shape(0)
		val n = shape(1)
		val nStorage = Pointer()
		cudaMalloc(nStorage, Sizeof.FLOAT * shape.product)
		val kernelParams = Pointer.to(
			Pointer.to(Array(shape(0))),
			Pointer.to(Array(shape(1))),
			Pointer.to(storage),
			Pointer.to(nStorage)
		)
		val bsx = if mBlockSize > m then m else mBlockSize
		val gsx = (m + mBlockSize - 1) / mBlockSize
		val bsy = if mBlockSize > n then n else mBlockSize
		val gsy = (n + mBlockSize - 1) / mBlockSize
		cernelExecute("src/main/resources/util.ptx", "matrixTransposition", kernelParams, gridDimX = gsy, gridDimY = gsx, blockDimX = bsy, blockDimY = bsx)
		new CudaStorage(nStorage, shape.reverse)

	def item = 
		if shape != Seq(1) then 
			throw new Exception("It is impossible to take an element from the storage if shape != Seq(1)")
		val item = Array(0f)
		cudaMemcpy(Pointer.to(item), storage, Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToHost)
		item(0)

	def unary_- = this * -1

	def pow(n: Float): Storage = 
		elementwiseScalarOperation(n, elementwiseCernelPath, "tensorPow")

	def cat(st: Storage, dim: Int = 0) = 
		val mBlockSize = 1024
		st match
			case st: CudaStorage =>
				if dim < 0 || dim >= shape.length then
					throw new Exception(s"Dim problem in cat:\ndim: $dim")
				if shape.patch(dim, Nil, 1) != st.shape.patch(dim, Nil, 1) then
					throw new Exception(s"Shape problem in cat: \nshape1: $shape, shape2: ${st.shape}")
				
				val xSize = shape.drop(dim).product
				val ySize = st.shape.drop(dim).product

				val nStorage = Pointer()
				val nShape = shape.updated(dim, shape(dim) + st.shape(dim))
				val m = nShape.product
				cudaMalloc(nStorage, Sizeof.FLOAT * m)
				val kernelParams = Pointer.to(
					Pointer.to(Array(m)),
					Pointer.to(storage),
					Pointer.to(st.storage),
					Pointer.to(Array(xSize)),
					Pointer.to(Array(ySize)),
					Pointer.to(nStorage)
				)
				val bs = if mBlockSize > m then m else mBlockSize
				val gs = (m + mBlockSize - 1) / mBlockSize
				cernelExecute("src/main/resources/util.ptx", "cat", kernelParams, gridDimX = gs, blockDimX = bs)  
				new CudaStorage(nStorage, nShape)
			case _ => throw new Exception("Can't cat not same type storages")


object CudaStorage:
	def apply(storage: Iterable[Float], shape: Seq[Int]) =
		require(storage.nonEmpty, "ArrayStorage: storage array must not be empty")
		require(shape != Seq(), "ArrayStorage: shape must not be empty")
		require(storage.toSeq.length == shape.product, "ArrayStorage: size of the data must correspond to shape")
		require(shape.map(_ > 0).reduce(_ && _), "ArrayStorage: shape must not contain dim < 0")
		val pointer = host2device(storage, shape.product)
		new CudaStorage(pointer, shape)
	def fill(shape: Seq[Int], value: =>Float): Storage = 
		val newStorage = Array.fill(shape.product)(value)
		CudaStorage(newStorage, shape)
	def ones(shape: Seq[Int]): Storage = fill(shape, 1)
	def zeros(shape: Seq[Int]): Storage = fill(shape, 0)
	def rand(shape: Seq[Int]): Storage = 
		val r = Random()
		CudaStorage.fill(shape, r.nextFloat())