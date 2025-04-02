package scuda.Tensor

import jcuda.Pointer
import jcuda.runtime._
import jcuda.runtime.JCuda._
import jcuda.Sizeof
import jcuda.jcublas.*


class CudaStorage(
	val storage: Pointer, 
	val shape: Seq[Int], 
) extends Storage:
	private def typeSize = 4
	
	override def toString(): String = beautifulArrayprint(device2host(storage, shape), shape)

	override def finalize() = cudaFree(storage)
	
	def +(other: Storage) =
			if this.shape != other.shape then throw new Exception("Operation cannot be performed if shape of Tensors isn't equal.")
			other match
					case _: ArrayStorage => throw new Exception("Operation cannot be performed if devise isn't same.")
					case other: CudaStorage => {
							val res = Pointer()
							cudaMalloc(res, shape.product * Sizeof.FLOAT)
							cudaMemcpy(res, other.storage, shape.product * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToDevice)
							
							val handle = new cublasHandle()
							JCublas2.cublasCreate(handle)
							JCublas2.cublasSaxpy(handle, shape.product, Pointer.to(Array(1.0f)), storage, 1, res, 1)
							
							new CudaStorage(res, shape)
					}
	
	def -(other: Storage): Storage = this + ( other * -1f )
	
	def *(other: Storage): Storage = 
			if this.shape != other.shape then throw new Exception("Operation cannot be performed if shape of Tensors isn't equal.")
			other match
					case _: ArrayStorage => throw new Exception("Operation cannot be performed if devise isn't same.")
					case other: CudaStorage => {
							val res = Pointer()
							cudaMalloc(res, shape.product * Sizeof.FLOAT)

							val handle = new cublasHandle()
							JCublas2.cublasCreate(handle)
							JCublas2.cublasSdgmm(handle, cublasSideMode.CUBLAS_SIDE_LEFT, 
									shape.product, 1, 
									this.storage, shape.product, 
									other.storage, 1, 
									res, shape.product
							)
							new CudaStorage(res, shape)
					}

	def /(other: Storage): Storage = 
			if this.shape != other.shape then 
					throw new Exception("Operation cannot be performed if shape of Tensors isn't equal.")

			other match
					case _: ArrayStorage => 
							throw new Exception("Operation cannot be performed if device isn't the same.")
					case other: CudaStorage => 
							val size = shape.product
							val d_invB = new Pointer()

							cudaMalloc(d_invB, size * Sizeof.FLOAT)

							val handle = new cublasHandle()
							JCublas2.cublasCreate(handle)

							// fill array invB by 1 / B
							val h_invB = new Array[Float](size)
							cudaMemcpy(Pointer.to(h_invB), other.storage, size * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToHost)
							if h_invB.contains(0) then throw new Exception("/ by zero")
							cudaMemcpy(d_invB, Pointer.to(h_invB.map(1 / _)), size * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice)
							
							this * new CudaStorage(d_invB, shape)

	def +(alpha: Float): Storage = 
			val res = Pointer()
			cudaMalloc(res, shape.product * Sizeof.FLOAT)
			cudaMemcpy(res, storage, shape.product * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToDevice)

			val ones = Array.fill[Float](shape.product)(1.0f)
			val d_ones = Pointer()
			cudaMalloc(d_ones, shape.product * Sizeof.FLOAT)
			cudaMemcpy(d_ones, Pointer.to(ones), shape.product * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice)
			
			val handle = new cublasHandle()
			JCublas2.cublasCreate(handle)
			// y = alpha * x + y
			JCublas2.cublasSaxpy(handle, shape.product, Pointer.to(Array(alpha)), d_ones, 1, res, 1)
			
			new CudaStorage(res, shape)
	def -(alpha: Float): Storage = this + ( -alpha )

	def *(alpha: Float): Storage = 
			val res = Pointer()
			cudaMalloc(res, shape.product * Sizeof.FLOAT)
			cudaMemcpy(res, storage, shape.product * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToDevice)

			val handle = new cublasHandle()
			JCublas2.cublasCreate(handle)
			JCublas2.cublasSscal(handle, shape.product, Pointer.to(Array(alpha)), res, 1)
			new CudaStorage(res, shape)

	def /(alpha: Float): Storage = this * (1 / alpha)
			
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
		// todo: realization with CUDA power
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

