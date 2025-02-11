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





trait Storage:
    val shape: Seq[Int]
    override def toString(): String
    def +(other: Storage): Storage
    def -(other: Storage): Storage
    def *(other: Storage): Storage
    def *(alpha: Float): Storage
    def /(other: Storage): Storage
    

class ArrayStorage(
    val storage: Array[Float], 
    val shape: Seq[Int]
    ) extends Storage:
    override def toString(): String = beautifulArrayprint(storage, shape)

    def +(other: Storage) = elementByElementOperation(_ + _)(other)
    
    def -(other: Storage) = elementByElementOperation(_ - _)(other)
    
    def *(other: Storage) = elementByElementOperation(_ * _)(other)
    
    def *(alpha: Float): Storage = 
        val newData = storage.map(_ * alpha)
        new ArrayStorage(newData, shape)

    def /(other: Storage) = elementByElementOperation(_ / _)(other)

    def elementByElementOperation(operation: (Float, Float) => Float)(other: Storage): Storage = 
        if this.shape != other.shape then throw new Exception("Operation cannot be performed if shape of Tensors isn't equal.")
        other match
            case other: ArrayStorage => ArrayStorage(storage.zip(other.storage).map(operation(_,_)), shape)
            case _: CudaStorage  => throw new Exception("Operation cannot be performed if devise isn't same.")
               
    


class CudaStorage(
    val storage: Pointer, 
    val shape: Seq[Int], 
) extends Storage:
    private def typeSize = 4
    
    override def toString(): String = beautifulArrayprint(device2hostCopy(storage, shape), shape)

    override def finalize() = cudaFree(storage)
    

    def device2hostCopy(d_array: Pointer, shape: Seq[Int]) = {
        if shape(0) == 0 then new Array[Float](0)
        else
            val size = shape.product
            val h_array = new Array[Float](shape.product)
            val pointer = Pointer.to(h_array)
            cudaMemcpy(pointer, d_array, size * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToHost)
            h_array
    }

    def device2hostTransfer() =
        val newStorage = device2hostCopy(storage, this.shape)
        val shape = this.shape

        this.finalize()
        
        ArrayStorage(newStorage, shape)

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
    def *(other: Storage): Storage = ???

    def *(alpha: Float): Storage = 
        val res = Pointer()
        cudaMalloc(res, shape.product * Sizeof.FLOAT)
        cudaMemcpy(res, storage, shape.product * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToDevice)

        val handle = new cublasHandle()
        JCublas2.cublasCreate(handle)
        JCublas2.cublasSscal(handle, shape.product, Pointer.to(Array(alpha)), res, 1)
        new CudaStorage(res, shape)

    def /(other: Storage): Storage = ???
        

object CudaStorage:
    def host2device(seq: Seq[Float], len: Int) = 
        val h_array = seq.toArray
        val d_array = Pointer()
        cudaMalloc(d_array, len * Sizeof.FLOAT)
        cudaMemcpy(d_array, Pointer.to(h_array), len * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice)
        d_array
    def apply(h_array: Seq[Float], shape: Seq[Int]) =
        val pointer = host2device(h_array, shape.product)
        new CudaStorage(pointer, shape)


def beautifulArrayprint[T](storage: Array[T], shape: Seq[Int]) = {
    // all rows creating
    val last_dim = shape(shape.length - 1)
    var rows = storage.grouped(last_dim).map("[" + _.mkString(", ") + "]")
    
    for ((v,n) <- shape.zip((0 until shape.length).reverse).reverse.tail){
        rows = rows.grouped(v).map("[" + _.reduce(_ + "," + ("\n" * n) + _ ) + "]")
    }
    rows.reduce(_+_)         
}