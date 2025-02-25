package scuda.Tensor

import jcuda.Pointer
import jcuda.runtime._
import jcuda.runtime.JCuda._
import jcuda.Sizeof
import jcuda.jcublas.*


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

implicit class FloatOps(v: Float):
	def /(obj: Storage): Storage =
		Storage.fill(obj, v) / obj
	def *(obj: Storage): Storage =
		obj * v