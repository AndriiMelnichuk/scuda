package scuda.tensor.cuda

import scuda.tensor.storage.Storage

import jcuda.Pointer
import jcuda.Sizeof
import jcuda.runtime.JCuda._
import jcuda.driver.JCudaDriver._
import jcuda.runtime.cudaMemcpyKind
import jcuda.driver.CUmodule
import jcuda.driver.CUfunction

/**
 * Runs cuda kernels implementing element-by-element operations on 2 tensors
 *
 * @param a CudaStorage 1
 * @param b CudaStorage 2
 * @param ptxFile path to compiled cuda file
 * @param opName name of operation
 * @param mBlockSize max count of threads in block
 * @return operation result on CudaStorage
 */
def elementwiseCernelExecuter(a: CudaStorage, b: CudaStorage, ptxFile: String, opName: String, mBlockSize: Int = 1024): CudaStorage =    
	val d_a = a.storage
	val d_b = b.storage
	val d_c = Pointer()

	cudaMalloc(d_c, a.shape.product * Sizeof.FLOAT)

	// module load
	val module = CUmodule()
	cuModuleLoad(module, ptxFile)
	val function = CUfunction()
	cuModuleGetFunction(function, module, opName)

	val kernelParams = Pointer.to(
		Pointer.to(Array(a.shape.product)),
		Pointer.to(d_a),
		Pointer.to(d_b),
		Pointer.to(d_c)
	)
	val blockSize = if (mBlockSize > a.shape.product) a.shape.product else mBlockSize
	val gridSize = (a.shape.product + mBlockSize - 1) / mBlockSize
	cuLaunchKernel(function, gridSize, 1, 1, 
													blockSize, 1, 1, 
													0, null, kernelParams, null);
	new CudaStorage(d_c, a.shape)

def elementwiseScalarCernelExecuter(a: CudaStorage, b: Float, ptxFile: String, opName: String, mBlockSize: Int = 1024): CudaStorage =    
	val d_a = a.storage
	val d_c = Pointer()

	cudaMalloc(d_c, a.shape.product * Sizeof.FLOAT)

	// module load
	val module = CUmodule()
	cuModuleLoad(module, ptxFile)
	val function = CUfunction()
	cuModuleGetFunction(function, module, opName)

	val kernelParams = Pointer.to(
		Pointer.to(Array(a.shape.product)),
		Pointer.to(d_a),
		Pointer.to(Array(b)),
		Pointer.to(d_c)
	)
	val blockSize = if (mBlockSize > a.shape.product) a.shape.product else mBlockSize
	val gridSize = (a.shape.product + mBlockSize - 1) / mBlockSize
	cuLaunchKernel(function, gridSize, 1, 1, 
													blockSize, 1, 1, 
													0, null, kernelParams, null);
	new CudaStorage(d_c, a.shape)

def reducer(a: CudaStorage, ptxFile: String, opName: String): CudaStorage =
	// TODO how to choose blockCount and threadCount?
	val blockCount = 10 
	val threadCount = 100

	val d_a = a.storage
	val d_b = Pointer()
	val res = Pointer()
	cudaMalloc(d_b, a.shape.product * Sizeof.FLOAT)
	cudaMalloc(res, Sizeof.FLOAT)

	// module load
	val module = CUmodule()
	cuModuleLoad(module, ptxFile)
	val function = CUfunction()
	cuModuleGetFunction(function, module, opName)
	
	// step 1
	val kernelParams1 = Pointer.to(
		Pointer.to(Array(a.shape.product)),
		Pointer.to(d_a),
		Pointer.to(d_b)
	)
	cuLaunchKernel(function, blockCount, 1, 1, 
													threadCount, 1, 1, 
													0, null, kernelParams1, null);

	// step2
	val kernelParams2 = Pointer.to(
		Pointer.to(Array(blockCount)),
		Pointer.to(d_b),
		Pointer.to(res)
	)
	cuLaunchKernel(function, 1, 1, 1, 
													blockCount, 1, 1, 
													0, null, kernelParams2, null);
	new CudaStorage(res, Seq.fill(a.shape.length)(1))

	
def cernelExecute(ptxFile: String, opName: String, kernelParams: Pointer, 
									gridDimX: Int = 1,  gridDimY: Int = 1,  gridDimZ: Int = 1, 
									blockDimX: Int = 1, blockDimY: Int = 1, blockDimZ: Int = 1): Unit =
	// module load
	val module = CUmodule()
	cuModuleLoad(module, ptxFile)
	val function = CUfunction()
	cuModuleGetFunction(function, module, opName)
	cuLaunchKernel(function, gridDimX, gridDimY, gridDimZ, 
														blockDimX, blockDimY, blockDimZ,
														0, null, kernelParams, null);


def relu(x: CudaStorage, mBlockSize: Int = 1024): CudaStorage =
	val nStorage = Pointer()
	cudaMalloc(nStorage, Sizeof.FLOAT * x.shape.product)
	val kernelParams = Pointer.to(
		Pointer.to(Array(x.shape.product)),
		Pointer.to(x.storage),
		Pointer.to(nStorage)
	)
	val blockSize = if (mBlockSize > x.shape.product) x.shape.product else mBlockSize
	val gridSize = (x.shape.product + mBlockSize - 1) / mBlockSize
	cernelExecute("src/main/resources/activationFunction.ptx", "ReLU", kernelParams, gridDimX = gridSize, blockDimX = blockSize)
	new CudaStorage(nStorage, x.shape)

def reluGrad(x: CudaStorage, cg: CudaStorage, mBlockSize: Int = 1024): CudaStorage =
	val nStorage = Pointer()
	cudaMalloc(nStorage, Sizeof.FLOAT * x.shape.product)
	val kernelParams = Pointer.to(
		Pointer.to(Array(x.shape.product)),
		Pointer.to(x.storage),
		Pointer.to(cg.storage),
		Pointer.to(nStorage)
	)
	val blockSize = if (mBlockSize > x.shape.product) x.shape.product else mBlockSize
	val gridSize = (x.shape.product + mBlockSize - 1) / mBlockSize
	cernelExecute("src/main/resources/activationFunction.ptx", "ReLUGrad", kernelParams, gridDimX = gridSize, blockDimX = blockSize)
	new CudaStorage(nStorage, x.shape)

def sigmoid(x: CudaStorage, mBlockSize: Int = 1024): CudaStorage =
	val nStorage = Pointer()
	cudaMalloc(nStorage, Sizeof.FLOAT * x.shape.product)
	val kernelParams = Pointer.to(
		Pointer.to(Array(x.shape.product)),
		Pointer.to(x.storage),
		Pointer.to(nStorage)
	)
	val blockSize = if (mBlockSize > x.shape.product) x.shape.product else mBlockSize
	val gridSize = (x.shape.product + mBlockSize - 1) / mBlockSize
	cernelExecute("src/main/resources/activationFunction.ptx", "sigmoid", kernelParams, gridDimX = gridSize, blockDimX = blockSize)
	new CudaStorage(nStorage, x.shape)

def tanh(x: CudaStorage, mBlockSize: Int = 1024): CudaStorage =
	val nStorage = Pointer()
	cudaMalloc(nStorage, Sizeof.FLOAT * x.shape.product)
	val kernelParams = Pointer.to(
		Pointer.to(Array(x.shape.product)),
		Pointer.to(x.storage),
		Pointer.to(nStorage)
	)
	val blockSize = if (mBlockSize > x.shape.product) x.shape.product else mBlockSize
	val gridSize = (x.shape.product + mBlockSize - 1) / mBlockSize
	cernelExecute("src/main/resources/activationFunction.ptx", "tanhaf", kernelParams, gridDimX = gridSize, blockDimX = blockSize)
	new CudaStorage(nStorage, x.shape)

def crossEntropyLoss(pr: CudaStorage, target: CudaStorage, mBlockSize: Int = 1024): CudaStorage =
	val nStorage = Pointer()
	cudaMalloc(nStorage, Sizeof.FLOAT * target.shape.product)
	
	val kernelParams = Pointer.to(
		Pointer.to(Array(pr.shape(0))),
		Pointer.to(Array(pr.shape(1))),
		Pointer.to(pr.storage),
		Pointer.to(target.storage),
		Pointer.to(nStorage)
	)
	val blockSize = if (mBlockSize > pr.shape(0)) pr.shape(0) else mBlockSize
	val gridSize = (pr.shape(0) + mBlockSize - 1) / mBlockSize
	cernelExecute("src/main/resources/lossFunction.ptx", "crossEntropyLoss", kernelParams, gridDimX = gridSize, blockDimX = blockSize)
	new CudaStorage(nStorage, target.shape)

def crossEntropyLossGrad(pr: CudaStorage, target: CudaStorage, chainGrad: CudaStorage, mBlockSize: Int = 1024): CudaStorage =
	val nStorage = Pointer()
	val m = pr.shape(0)
	val n = pr.shape(1)
	cudaMalloc(nStorage, Sizeof.FLOAT * pr.shape.product)

	val kernelParams = Pointer.to(
		Pointer.to(Array(m)),
		Pointer.to(Array(n)),
		Pointer.to(pr.storage),
		Pointer.to(target.storage),
		Pointer.to(chainGrad.storage),
		Pointer.to(nStorage)
	)
	
	val blockSize = if (mBlockSize > pr.shape(0)) pr.shape(0) else mBlockSize
	val gridSize = (pr.shape(0) + mBlockSize - 1) / mBlockSize
	cernelExecute("src/main/resources/lossFunction.ptx", "crossEntropyLossGrad", kernelParams, gridDimX = gridSize, blockDimX = blockSize)  
	new CudaStorage(nStorage, pr.shape)

def stableSoftmax(x: CudaStorage, mBlockSize: Int = 1024): CudaStorage =
	val nStorage = Pointer()
	val m = x.shape(0)
	val n = x.shape(1)
	cudaMalloc(nStorage, Sizeof.FLOAT * m * n)

	val kernelParams = Pointer.to(
		Pointer.to(Array(m)),
		Pointer.to(Array(n)),
		Pointer.to(x.storage),
		Pointer.to(nStorage)
	)

	val blockSize = if mBlockSize > m then m else mBlockSize
	val gridSize = (m + mBlockSize - 1) / mBlockSize
	cernelExecute("src/main/resources/activationFunction.ptx", "stableSoftmax", kernelParams, gridDimX = gridSize, blockDimX = blockSize)  
	new CudaStorage(nStorage, x.shape)

def stableSoftmaxGrad(sm: CudaStorage, cg: CudaStorage, mBlockSize: Int = 1024) = 
	val nStorage = Pointer()
	val m = sm.shape(0)
	val n = sm.shape(1)
	cudaMalloc(nStorage, Sizeof.FLOAT * m * n)

	val kernelParams = Pointer.to(
		Pointer.to(Array(m)),
		Pointer.to(Array(n)),
		Pointer.to(sm.storage),
		Pointer.to(cg.storage),
		Pointer.to(nStorage)
	)

	val bsx = if mBlockSize > m then m else mBlockSize
	val gsx = (m + mBlockSize - 1) / mBlockSize
	val bsy = if mBlockSize > n then n else mBlockSize
	val gsy = (n + mBlockSize - 1) / mBlockSize
	cernelExecute("src/main/resources/activationFunction.ptx", "stableSoftmaxGrad", kernelParams, gridDimX = gsx, gridDimY = gsy, blockDimX = bsx, blockDimY = bsy)  
	new CudaStorage(nStorage, sm.shape)

def broadcasting(x: CudaStorage, shape: Seq[Int], mBlockSize: Int = 1024): CudaStorage =
	def shapeCheck(rs: Seq[Int], es: Seq[Int]): Unit = 
		if es.length > rs.length then
			val res = Seq.fill(es.length - rs.length)(1) concat rs
			shapeCheck(res, es)
		else if es.length == rs.length then
			(es zip rs).foreach{ (e, r) =>
				if r > e then
					throw new Exception(s"broadcasting: real shape > expected shape. rs: $rs, es: $es")
				else if r != 1 && r != e then 
					throw new Exception(s"broadcasting: expected shape: $e, real shape: $r")
			}
		else 
			throw new Exception(s"broadcasting: real shape len > expected shape len. rs: $rs, es: $es")
	shapeCheck(x.shape, shape)
	def recursiveBroadcasting(a: Pointer, n: Int): CudaStorage = 
		if n == -1                     then new CudaStorage(a, shape)
		else if x.shape(n) == shape(n) then recursiveBroadcasting(a, n - 1)
		else
			val elems2copy = shape.drop(n + 1).product
			val copyCount  = shape(n)
			val nShape     = (0 until shape.length).map{i => if i < n then x.shape(i) else shape(i)}
			val nStorage   = Pointer()
			cudaMalloc(nStorage, Sizeof.FLOAT * nShape.product)
			val kernelParams = Pointer.to(
				Pointer.to(Array(elems2copy)),
				Pointer.to(Array(copyCount)),
				Pointer.to(Array(nShape.product)),
				Pointer.to(a),
				Pointer.to(nStorage)
			)
			val blockSize = if (mBlockSize > nShape.product) nShape.product else mBlockSize
			val gridSize = (nShape.product + mBlockSize - 1) / mBlockSize
			cernelExecute("src/main/resources/util.ptx", "broadcasting", kernelParams, gridDimX = gridSize, blockDimX = blockSize)
			if a != x.storage then 
				cudaFree(a)
			recursiveBroadcasting(nStorage, n - 1)
	recursiveBroadcasting(x.storage, shape.length - 1)