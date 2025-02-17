package scuda

import scala.math.{Fractional, Numeric}
import scala.reflect.ClassTag

import jcuda.Pointer
import jcuda.runtime._
import jcuda.runtime.JCuda._
import jcuda.Sizeof
import jcuda.jcublas.*

import java.lang.instrument.Instrumentation
import scala.annotation.internal.sharable
import scala.annotation.alpha
import scala.compiletime.ops.float

import scala.collection.parallel.CollectionConverters._

class Tensor(computeStorage: () => Storage):
    lazy val storage: Storage = computeStorage()
    override def toString(): String = storage.toString()
    def +(other: Tensor) = new Tensor(() => this.storage + other.storage)
    def -(other: Tensor) = new Tensor(() => this.storage - other.storage)
    def *(other: Tensor) = new Tensor(() => this.storage * other.storage)
    def /(other: Tensor) = new Tensor(() => this.storage / other.storage)
    def **(other: Tensor) = new Tensor(() => this.storage ** other.storage)
    
    def +(alpha: Float) = new Tensor(() => this.storage + alpha)
    def -(alpha: Float) = new Tensor(() => this.storage - alpha)
    def *(alpha: Float) = new Tensor(() => this.storage * alpha)
    def /(alpha: Float) = new Tensor(() => this.storage / alpha)
    
    def toCpu(): Tensor = new Tensor(() => this.storage.toCpu())
    def toCuda(): Tensor = new Tensor(() => this.storage.toCuda())

object Tensor:
    def apply(data: Seq[Float]) = new Tensor(() => ArrayStorage(data.toArray, Seq(data.length)))

    def apply(data: Seq[Float], shape: Seq[Int]) = new Tensor(() => {
        if shape.product != data.length then throw new Exception("Elligal shape")
        if shape.isEmpty then throw new Exception("Shape empty")
        ArrayStorage(data.toArray, shape)
    })

    def apply(data: Seq[Float], storageType: String) = new Tensor(() => {
        storageType.toLowerCase match
            case "cpu" => ArrayStorage(data.toArray, Seq(data.length))
            case "cuda" => CudaStorage(data.toArray, Seq(data.length))
            case _ => throw new Exception("Unknown device type")
    })

    def apply(data: Seq[Float], shape: Seq[Int], storageType: String) = new Tensor(() => {
        if shape.product != data.length then throw new Exception("Elligal shape")
        if shape.isEmpty then throw new Exception("Shape empty")
        storageType.toLowerCase match
            case "cpu" => ArrayStorage(data.toArray, shape)
            case "cuda" => CudaStorage(data.toArray, shape)
            case _ => throw new Exception("Unknown device type")
    })
    

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
    
    def toCpu(): ArrayStorage
    def toCuda(): CudaStorage
    
class ArrayStorage(
    val storage: Array[Float], 
    val shape: Seq[Int]
    ) extends Storage:
    override def toString(): String = beautifulArrayprint(storage, shape)

    def +(other: Storage) = elementByElementOperation(_ + _)(other)
    
    def -(other: Storage) = elementByElementOperation(_ - _)(other)
    
    def *(other: Storage) = elementByElementOperation(_ * _)(other)
    
    def /(other: Storage) = elementByElementOperation(_ / _)(other)

    def +(alpha: Float) = new ArrayStorage(storage.map(_ + alpha), shape)

    def -(alpha: Float) = new ArrayStorage(storage.map(_ - alpha), shape)

    def *(alpha: Float): Storage = new ArrayStorage(storage.map(_ * alpha), shape)

    def /(alpha: Float): Storage = new ArrayStorage(storage.map(_ / alpha), shape)    

    def **(other: Storage): Storage = 
        if this.shape.length != 2 || other.shape.length != 2 then throw new Exception("Operation cannot be performed if shape of Tensors isn't equal 2.")
        if this.shape(1) != other.shape(0) then throw new Exception("Operation cannot be performed if Tensor A.shape(1) != Tensor B.shape(0)")

        other match
            case _: CudaStorage  => throw new Exception("Operation cannot be performed if devise isn't same.")
            case other: ArrayStorage => 
                val m = this.shape(0)
                val k = this.shape(1)
                val n = other.shape(1)

                val A = this.storage
                val B = other.storage

                val nStorage = (0 until m * n).map( i => (i / n, i % n))
                    .map( (i,j) => ((0 until k).map(_ + i * k),  (0 until k).map(_ *n + j)))
                    .map( (a_ind, b_ind) => (a_ind.map(A(_)) zip b_ind.map(B(_))).map(_*_).sum)

                ArrayStorage(nStorage.toArray, Seq(m, n))

    def elementByElementOperation(operation: (Float, Float) => Float)(other: Storage): Storage = 
        if this.shape != other.shape then throw new Exception("Operation cannot be performed if shape of Tensors isn't equal.")
        other match
            case other: ArrayStorage => ArrayStorage(storage.zip(other.storage).map(operation(_,_)), shape)
            case _: CudaStorage  => throw new Exception("Operation cannot be performed if devise isn't same.")
    
    def toCpu(): ArrayStorage = new ArrayStorage(storage.clone(), shape)

    def toCuda(): CudaStorage = new CudaStorage(host2device(storage, shape.product), shape)

    


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

                // Заполняем массив invB значениями 1 / B
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

object CudaStorage:

    def apply(h_array: Seq[Float], shape: Seq[Int]) =
        val pointer = host2device(h_array, shape.product)
        new CudaStorage(pointer, shape)

def beautifulArrayprint[T](storage: Array[T], shape: Seq[Int]): String = 
    if storage.isEmpty then return "[]"
    
    // all rows creating
    val last_dim = shape(shape.length - 1)
    var rows = storage.grouped(last_dim).map("[" + _.mkString(", ") + "]")
    
    for ((v,n) <- shape.zip((0 until shape.length).reverse).reverse.tail){
        rows = rows.grouped(v).map("[" + _.reduce(_ + "," + ("\n" * n) + _ ) + "]")
    }
    rows.reduce(_+_)         

def host2device(seq: Seq[Float], len: Int) = 
    val h_array = seq.toArray
    val d_array = Pointer()
    cudaMalloc(d_array, len * Sizeof.FLOAT)
    cudaMemcpy(d_array, Pointer.to(h_array), len * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice)
    d_array

def device2host(d_array: Pointer, shape: Seq[Int]) = 
    if shape(0) == 0 then new Array[Float](0)
    else
        val size = shape.product
        val h_array = new Array[Float](shape.product)
        val pointer = Pointer.to(h_array)
        cudaMemcpy(pointer, d_array, size * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToHost)
        h_array