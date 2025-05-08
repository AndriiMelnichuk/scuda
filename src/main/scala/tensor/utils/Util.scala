package scuda.tensor.utils

import scuda.tensor.storage.Storage

import jcuda.Pointer
import jcuda.runtime._
import jcuda.runtime.JCuda._
import jcuda.Sizeof
import jcuda.jcublas.*
import scuda.tensor.Tensor

import scala.io.Source
import scala.collection.parallel.CollectionConverters._

def beautifulArrayprint[T](storage: Array[T], shape: Seq[Int]): String = 
	if storage.isEmpty then return "[]"
	
	// all rows creating
	val last_dim = shape(shape.length - 1)
	var rows = storage.grouped(last_dim).map("[" + _.mkString(", ") + "]")
	
	for ((v,n) <- shape.zip((0 until shape.length).reverse).reverse.tail){
			rows = rows.grouped(v).map("[" + _.reduce(_ + "," + ("\n" * n) + _ ) + "]")
	}
	rows.reduce(_+_)         

def host2device(seq: Iterable[Float], len: Int) = 
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

def readCsv(path: String, isHeader: Boolean = false, splitSign: String = ",", isGrad: Boolean = false)(using device: String = "cpu"): Tensor =
	var res = Source.fromFile(path)
		.getLines()
		.toArray
		.par
		.map(line => line.trim().split(splitSign))

	val data = if isHeader then res.tail else res
	val m = data.length
	val n = data(0).length

	data.foreach{line =>
		if line.length != n then throw new Exception(s"readCsv: different lenght of rows in file: $path.")
	}


	Tensor(data.flatten.map(_.toFloat).toArray, Seq(m, n), isGrad)

def readImg(path: String, isGrad: Boolean = false)(using device: String = "cpu") =
	import java.io.File
	import javax.imageio.ImageIO
	import java.awt.image.BufferedImage

	val img: BufferedImage = ImageIO.read(new File(path))

	val w = img.getWidth
	val h = img.getHeight
	val imgArray = Array.ofDim[Float](w * h * 3)

	(0 until w * h)
	.par
	.foreach{ i =>
		val x = i % w
		val y = i / w

		val rgb = img.getRGB(x, y)
		val r = (rgb >> 16) & 0xFF
		val g = (rgb >> 8) & 0xFF
		val b = rgb & 0xFF	

		imgArray(i)             = r
		imgArray(i + w * h)     = g
		imgArray(i + w * h * 2) = b
	}

	Tensor(imgArray, Seq(3, h, w), isGrad) / 255


def reverseBroadcasting(x: Storage, s: Seq[Int]): Storage = 
  def helper(x: Storage, ax: Seq[Int]): Storage = 
    ax match
      case Nil => x
      case h :: t =>
        val newAx = s.map(_ - 1)
        val newStorage = x.sum(h)
        helper(newStorage, t)
  if x.shape == s then x
  else
    val axes = (0 until x.shape.length).filter(ax => x.shape(ax) != s(ax))
    helper(x, axes)


  







