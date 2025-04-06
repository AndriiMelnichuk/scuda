package scuda.tensor.cuda

import scuda.tensor.Storage
import scuda.tensor.cpu.ArrayStorage
import scuda.tensor.utils.{ beautifulArrayprint, device2host, host2device }
import scala.collection.parallel.CollectionConverters._

import jcuda.Pointer
import jcuda.runtime._
import jcuda.runtime.JCuda._
import jcuda.Sizeof
import jcuda.jcublas.*
import scala.collection.View.FlatMap

// TODO change stule - case other: Cuda... , case _ =>.
// other match
// 		case _: ArrayStorage => throw new Exception("Operation cannot be performed if devise isn't same.")
// 		case other: CudaStorage => {
// 				val res = Pointer()
// 				cudaMalloc(res, shape.product * Sizeof.FLOAT)

// 				val handle = new cublasHandle()
// 				JCublas2.cublasCreate(handle)
// 				JCublas2.cublasSdgmm(handle, cublasSideMode.CUBLAS_SIDE_LEFT, 
// 						shape.product, 1, 
// 						this.storage, shape.product, 
// 						other.storage, 1, 
// 						res, shape.product
// 				)
// 				new CudaStorage(res, shape)

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
				case other: CudaStorage => CudaOperations.elementwiseCernelExecuter(this, other, ptxFile, opName)
				case _ => throw new Exception("Operation cannot be performed if devise isn't same.")

	def elementwiseScalarOperation(other: Float, ptxFile: String, opName: String): Storage = 
		CudaOperations.elementwiseScalarCernelExecuter(this, other, ptxFile, opName)

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
					case _: ArrayStorage => 
							throw new Exception("Operation cannot be performed if devise isn't same.")
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
		new CudaStorage(storage, shape.reverse)

	// TODO can be optimized
	def item = this.toCpu().item

object CudaStorage:

	def apply(h_array: Seq[Float], shape: Seq[Int]) =
			val pointer = host2device(h_array, shape.product)
			new CudaStorage(pointer, shape)

