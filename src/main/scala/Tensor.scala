package scuda

import scala.math.{Fractional, Numeric}
import scala.reflect.ClassTag

import jcuda.Pointer
import jcuda.runtime._
import jcuda.Sizeof
import jcuda.jcublas.*

import java.lang.instrument.Instrumentation
import scala.annotation.internal.sharable

class Tensor[T: Numeric :ClassTag](val storage: Either[Array[T], Pointer], val shape: Seq[Int], val data_type: ClassTag[T]) {
    
    override def toString(): String = {
        def helper(storage: Array[T]) = {
            // all rows creating
            val last_dim = shape(shape.length - 1)
            var rows = storage.grouped(last_dim).map("[" + _.mkString(", ") + "]")
            
            for ((v,n) <- shape.zip((0 until shape.length).reverse).reverse.tail){
                rows = rows.grouped(v).map("[" + _.reduce(_ + "," + ("\n" * n) + _ ) + "]")
            }
            rows.reduce(_+_)         
        }
        if shape(0) == 0 then "[]"
        else
            storage match
                case Left(value) => helper(value)
                case Right(value) => helper(device2host(value,shape))
                
            

    }

    def elementByElementOperation(operation: (T, T) => T)(other: Tensor[T]): Tensor[T] = {
        if this.shape != other.shape then throw new Exception("Operation cannot be performed if shape of Tensors isn't equal.")
        if this.device != other.device then throw new Exception("Operation cannot be performed if devise isn't same.")
        
        // TODO cuda realization
        val newArray = this.storage.zip(other.storage).map(operation(_,_))
        new Tensor(newArray, this.shape, device)
    }

    def +(other: Tensor[T]) = elementByElementOperation(implicitly[Numeric[T]].plus)(other)
    def -(other: Tensor[T]) = elementByElementOperation(implicitly[Numeric[T]].minus)(other)
    def *(other: Tensor[T]) = elementByElementOperation(implicitly[Numeric[T]].times)(other)
    
    // TODO: div with type change
    // def /(other: Tensor[T]) = elementByElementOperation()

}

object Tensor{
    def apply[T :Numeric :ClassTag](data: Seq[T], shape: Seq[Int], device: String) = {
        // argument correctness check
        if !data.isEmpty && shape.isEmpty then 
            objectConstructor(data, Seq(0), device)
        else if data.isEmpty && shape.isEmpty then
            objectConstructor(Seq(), Seq(0), device)
        else if !data.isEmpty && !shape.isEmpty then
            if data.length != shape.product then 
                throw new ArithmeticException(s"It's impossible to create a tensor from an array of length ${data.length}, with a shape for ${shape.sum}.")
            objectConstructor(data, shape, device)
        else
            throw new ArithmeticException(s"It's impossible to create a tensor from an array of length ${data.length}, with a shape for ${shape.sum}.")
    }

    private def objectConstructor[T :Numeric :ClassTag](data: Seq[T], shape: Seq[Int], device: String) = {
        if !List("cpu", "cuda").contains(device) then
            throw new java.lang.IllegalArgumentException(s"unknown device- $device")
        
        if device == "cpu" then
            new Tensor[T](Left(data.toArray), shape, implicitly[ClassTag[T]])
        else
            val h_array = data.toArray
            val size = getTypeSize(h_array)

            // TODO: I must del it
            JCuda.setExceptionsEnabled(true)
            JCuda.cudaSetDevice(0)

            val d_array = new Pointer()
            val pointer = getHostPointer(h_array)
            JCuda.cudaMalloc(d_array, data.length * size)
            JCuda.cudaMemcpy(d_array, pointer, data.length * size, cudaMemcpyKind.cudaMemcpyHostToDevice)
            
            new Tensor[T](Right(d_array), shape, implicitly[ClassTag[T]])
    }

    def apply[T :Numeric :ClassTag](data: Seq[T], shape: Seq[Int]): Tensor[T] = apply(data, shape, "cpu")
   
    def apply[T :Numeric :ClassTag](data: Seq[T]): Tensor[T] = apply(data, Seq(data.length), "cpu")
}

// useful functions

def host2device[T: Numeric :ClassTag](seq: Seq[T]) = {
    val h_array = seq.toArray
    val ct = implicitly[ClassTag[T]]
    val typeName = ct.runtimeClass.getSimpleName
    val size = typeName match {
        case "int" => 4
        case "long" => 8
        case "float" => 4
        case "double" => 8
        case "char" => 2
        case "byte" => 1
        case "boolean" => 1
        case _ => throw new IllegalArgumentException("Unsupported type")
    }

    JCuda.setExceptionsEnabled(true)
    JCuda.cudaSetDevice(0)

    val d_array = new Pointer()
    val pointer = typeName match {
        case "int"    => Pointer.to(h_array.asInstanceOf[Seq[Int]].toArray)
        case "long"   => Pointer.to(h_array.asInstanceOf[Seq[Long]].toArray)
        case "float"  => Pointer.to(h_array.asInstanceOf[Seq[Float]].toArray)
        case "double" => Pointer.to(h_array.asInstanceOf[Seq[Double]].toArray)
        case "char"   => Pointer.to(h_array.asInstanceOf[Seq[Char]].toArray)
        case "byte"   => Pointer.to(h_array.asInstanceOf[Seq[Byte]].toArray)
        case _        => throw new IllegalArgumentException("Unsupported type for CUDA")
    }
    JCuda.cudaMalloc(d_array, h_array.length * size)
    JCuda.cudaMemcpy(d_array, pointer, h_array.length * size, cudaMemcpyKind.cudaMemcpyHostToDevice)

    d_array
}

def device2host[T: ClassTag :Numeric](d_array: Pointer, shape: Seq[Int]) = {
    if shape(0) == 0 then new Array[T](0)
    else
        val size = shape.product
        val h_array = new Array[T](shape.product)
        val pointer = getHostPointer(h_array)

        val ct = implicitly[ClassTag[T]]
        ct match {
            case ClassTag.Int    => JCuda.cudaMemcpy(pointer, d_array, size * Sizeof.INT, cudaMemcpyKind.cudaMemcpyDeviceToHost)
            case ClassTag.Long   => JCuda.cudaMemcpy(pointer, d_array, size * Sizeof.LONG, cudaMemcpyKind.cudaMemcpyDeviceToHost)
            case ClassTag.Float  => JCuda.cudaMemcpy(pointer, d_array, size * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToHost)
            case ClassTag.Double => JCuda.cudaMemcpy(pointer, d_array, size * Sizeof.DOUBLE, cudaMemcpyKind.cudaMemcpyDeviceToHost)
            case ClassTag.Char   => JCuda.cudaMemcpy(pointer, d_array, size * Sizeof.CHAR, cudaMemcpyKind.cudaMemcpyDeviceToHost)
            case ClassTag.Byte   => JCuda.cudaMemcpy(pointer, d_array, size * Sizeof.BYTE, cudaMemcpyKind.cudaMemcpyDeviceToHost)
            case _               => throw new IllegalArgumentException("Unsupported type for CUDA")
        }
        // JCuda.cudaMemcpy()

        h_array
}

def getHostPointer[T :ClassTag :Numeric](data: Array[T]) = {
    val ct = implicitly[ClassTag[T]]
    ct match {
        case ClassTag.Int    => Pointer.to(data.asInstanceOf[Seq[Int]].toArray)
        case ClassTag.Long   => Pointer.to(data.asInstanceOf[Seq[Long]].toArray)
        case ClassTag.Float  => Pointer.to(data.asInstanceOf[Seq[Float]].toArray)
        case ClassTag.Double => Pointer.to(data.asInstanceOf[Seq[Double]].toArray)
        case ClassTag.Char   => Pointer.to(data.asInstanceOf[Seq[Char]].toArray)
        case ClassTag.Byte   => Pointer.to(data.asInstanceOf[Seq[Byte]].toArray)
        case _               => throw new IllegalArgumentException("Unsupported type for CUDA")
    }
}

def getTypeSize[T :ClassTag :Numeric](data: Array[T]) = {
    val ct = implicitly[ClassTag[T]]
    ct match {
        case ClassTag.Int    => 4
        case ClassTag.Long   => 8
        case ClassTag.Float  => 4
        case ClassTag.Double => 8
        case ClassTag.Char   => 2
        case ClassTag.Byte   => 1
        case _               => throw new IllegalArgumentException("Unsupported type for CUDA")
    }
}


trait Storage[T]:
    val shape: Seq[Int]
    override def toString(): String
    def +(other: Storage[T]): Storage[T]
    def -(other: Storage[T]): Storage[T]
    def *(other: Storage[T]): Storage[T]
    def /(other: Storage[T]): Storage[Double]
    

class ArrayStorage[T :Numeric :ClassTag](
    val storage: Array[T], 
    val shape: Seq[Int]
    ) extends Storage[T]:
    override def toString(): String = beautifulArrayPrint(storage, shape)

    def +(other: Storage[T]) = elementByElementOperation(implicitly[Numeric[T]].plus)(other)
    
    def -(other: Storage[T]) = elementByElementOperation(implicitly[Numeric[T]].minus)(other)
    
    def *(other: Storage[T]) = elementByElementOperation(implicitly[Numeric[T]].times)(other)
    
    def /(other: Storage[T]): Storage[Double] =
        if this.shape != other.shape then throw new Exception("Operation cannot be performed if shape of Tensors isn't equal.")
        other match
            case _: CudaStorage[T] => throw new Exception("Operation cannot be performed if device isn't same.")
            case other: ArrayStorage[T] =>
                ArrayStorage(storage.zip(other.storage).map {
                    case (v1, v2) =>
                        val numeric = implicitly[Numeric[T]]
                        val dividend = numeric.toDouble(v1)
                        val divisor = numeric.toDouble(v2)
                        dividend / divisor
                }, shape)

    def elementByElementOperation(operation: (T, T) => T)(other: Storage[T]): Storage[T] = 
        if this.shape != other.shape then throw new Exception("Operation cannot be performed if shape of Tensors isn't equal.")
        other match
            case other: ArrayStorage[T] => ArrayStorage(storage.zip(other.storage).map(operation(_,_)), shape)
            case _: CudaStorage[T]  => throw new Exception("Operation cannot be performed if devise isn't same.")
               
    


class CudaStorage[T :Numeric :ClassTag](
    val storage: Pointer, 
    val shape: Seq[Int], 
) extends Storage[T]:
    private def getTypeSize = 
        val ct = implicitly[ClassTag[T]]
        ct match {
            case ClassTag.Int    => 4
            case ClassTag.Long   => 8
            case ClassTag.Float  => 4
            case ClassTag.Double => 8
            case ClassTag.Char   => 2
            case ClassTag.Byte   => 1
            case _               => throw new IllegalArgumentException("Unsupported type for CUDA")
        }
    
    override def toString(): String = beautifulArrayPrint(device2hostCopy(storage, shape), shape)

    override def finalize() = JCuda.cudaFree(storage)
    

    def device2hostCopy[T: ClassTag :Numeric](d_array: Pointer, shape: Seq[Int]) = {
        if shape(0) == 0 then new Array[T](0)
        else
            val size = shape.product
            val h_array = new Array[T](shape.product)
            val pointer = getHostPointer(h_array)
    
            val ct = implicitly[ClassTag[T]]
            ct match {
                case ClassTag.Int    => JCuda.cudaMemcpy(pointer, d_array, size * Sizeof.INT, cudaMemcpyKind.cudaMemcpyDeviceToHost)
                case ClassTag.Long   => JCuda.cudaMemcpy(pointer, d_array, size * Sizeof.LONG, cudaMemcpyKind.cudaMemcpyDeviceToHost)
                case ClassTag.Float  => JCuda.cudaMemcpy(pointer, d_array, size * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToHost)
                case ClassTag.Double => JCuda.cudaMemcpy(pointer, d_array, size * Sizeof.DOUBLE, cudaMemcpyKind.cudaMemcpyDeviceToHost)
                case ClassTag.Char   => JCuda.cudaMemcpy(pointer, d_array, size * Sizeof.CHAR, cudaMemcpyKind.cudaMemcpyDeviceToHost)
                case ClassTag.Byte   => JCuda.cudaMemcpy(pointer, d_array, size * Sizeof.BYTE, cudaMemcpyKind.cudaMemcpyDeviceToHost)
                case _               => throw new IllegalArgumentException("Unsupported type for CUDA")
            }
            // JCuda.cudaMemcpy()
    
            h_array
    }

    def device2hostTransfer() =
        val newStorage = device2hostCopy(storage, shape)
        val shape = this.shape

        this.finalize()
        
        ArrayStorage(newStorage, shape)

    def +(other: Storage[T]) =
        if this.shape != other.shape then throw new Exception("Operation cannot be performed if shape of Tensors isn't equal.")
        other match
            case _: ArrayStorage[?] => throw new Exception("Operation cannot be performed if devise isn't same.")
            case other: CudaStorage[T] => {
                val dAlpha = Pointer()
                JCuda.cudaMalloc(dAlpha, Sizeof)

                val res = Pointer()

                JCuda.cudaMalloc(res, shape.product)
                JCuda.cudaMemcpy(res, other.storage, shape.product, cudaMemcpyKind.cudaMemcpyDeviceToDevice)
                val handle = new cublasHandle()
                JCublas2.cublasCreate(handle)
                JCublas2.cublasSaxpy(handle, shape.product, )
                ???
            }
            
        

object CudaStorage:
    def apply[T :Numeric :ClassTag](h_array: Seq[T], shape: Seq[Int]) =
        val pointer = host2device(h_array)
        new CudaStorage[T](pointer, shape)


def beautifulArrayPrint[T](storage: Array[T], shape: Seq[Int]) = {
    // all rows creating
    val last_dim = shape(shape.length - 1)
    var rows = storage.grouped(last_dim).map("[" + _.mkString(", ") + "]")
    
    for ((v,n) <- shape.zip((0 until shape.length).reverse).reverse.tail){
        rows = rows.grouped(v).map("[" + _.reduce(_ + "," + ("\n" * n) + _ ) + "]")
    }
    rows.reduce(_+_)         
}