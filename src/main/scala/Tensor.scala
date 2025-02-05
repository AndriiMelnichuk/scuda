package scuda

import scala.math.{Fractional, Numeric}
import scala.reflect.ClassTag

class Tensor[T: Numeric :ClassTag](val storage: Array[T], val shape: Seq[Int], val device: String) {
    
    override def toString(): String = {
        if storage.length == 0 then "[]"
        else
            // all rows creating
            val last_dim = shape(shape.length - 1)
            var rows = storage.grouped(last_dim).map("[" + _.mkString(", ") + "]")
            
            for ((v,n) <- shape.zip((0 until shape.length).reverse).reverse.tail){
                rows = rows.grouped(v).map("[" + _.reduce(_ + "," + ("\n" * n) + _ ) + "]")
            }
            rows.reduce(_+_)
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

    // def apply(i: Int): Tensor[T] = 
    // def item(): T = if this.shape(0) != 0 then storage(0) else None
}

object Tensor{
   def apply[T :Numeric :ClassTag](data: Seq[T], shape: Seq[Int], device: String) = {
    // TODO: data move to device

    if data.length == 0 then 
        new Tensor[T](Array(), Seq(), device)
    else
        // argument correctness check
        if data.length != shape.product then 
            throw new ArithmeticException(s"It's impossible to create a tensor from an array of length ${data.length}, with a shape for ${shape.sum}.")
        if !List("cpu", "cuda").contains(device) then
            throw new java.lang.IllegalArgumentException(s"unknown device- $device")

        new Tensor[T](data.toArray, shape, device)
   }
   
   def apply[T :Numeric :ClassTag](data: Seq[T], shape: Seq[Int]): Tensor[T] = apply(data, shape, "cpu")
}