package scuda.tensor.cpu

import scuda.tensor.storage.Storage
import scuda.tensor.cuda.CudaStorage
import scuda.tensor.utils.{ beautifulArrayprint, host2device }
import scala.collection.parallel.CollectionConverters._
import scala.util.Random

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
			case _  => throw new Exception("Operation cannot be performed if devise isn't same.")

	def elementByElementOperation(operation: (Float, Float) => Float)(other: Storage): Storage = 
		if this.shape != other.shape then throw new Exception("Operation cannot be performed if shape of Tensors isn't equal.")
		other match
			case other: ArrayStorage => ArrayStorage(storage.zip(other.storage).map(operation(_,_)), shape)
			case _ => throw new Exception("Operation cannot be performed if devise isn't same.")
	
	def toCpu(): ArrayStorage = new ArrayStorage(storage.clone(), shape)

	def toCuda(): CudaStorage = new CudaStorage(host2device(storage, shape.product), shape)

	def T: Storage = 
		if shape.length != 2 then throw new Exception("not 2d Tensor cant be transponed")
		val newStorage =
		for
			i <- 0 until shape(1)
			j <- 0 until shape(0)
		yield storage(j * shape(1) + i)

		ArrayStorage(newStorage.toArray, shape.reverse)

	def sum: Storage = ArrayStorage(Array(storage.par.reduce(_ + _)), Seq(1))

	def item: Float = 
		if shape == Seq(1) then storage(0)
		else throw new Exception("It is impossible to take an element from the storage if shape != Seq(1)")

	def unary_- = new ArrayStorage(storage.map(-_), shape)

	def pow(n: Float) =
		new ArrayStorage(storage.par.map(x => math.pow(x.toDouble,n).toFloat).toArray, shape)

	// TODO index selection
	def apply(args: Int | Iterable[Int]*) =
		def helper(index: Int, level: Int) = ???
		args.foreach{
			case i: Int => println(s"Int: $i")
			case it: Iterable[Int] => println(s"Iterable ${it.toList}")
			case _ => println("who knows")
			
		}
		???

	def cat(st: Storage, dim: Int = 0) = 
		st match
			case st: ArrayStorage => 	
				if dim < 0 || dim >= shape.length then
					throw new Exception(s"Dim problem in cat:\ndim: $dim")
				if shape.patch(dim, Nil, 1) != st.shape.patch(dim, Nil, 1) then
					throw new Exception(s"Shape problem in cat: \nshape1: $shape, shape2: ${st.shape}")
				
				val x = storage.toList
				val y = st.storage.toList

				val xSize = shape.drop(dim).product
				val ySize = st.shape.drop(dim).product

				val groupedX = (x grouped xSize).toList
				val groupedY = (y grouped ySize).toList

				val res = 0.until(groupedX.length + groupedY.length).par
				.map(i => if i % 2 == 0 then groupedX(i / 2) else groupedY(i / 2))
				.flatten.toArray

				ArrayStorage(res, shape.updated(dim, shape(dim) + st.shape(dim)))
			case _ => throw new Exception("Can't cat not same type storages")

object ArrayStorage:
	def apply(data: Iterable[Float], shape: Seq[Int]): ArrayStorage = 
		new ArrayStorage(data.toArray, shape)
	def fill(shape: Seq[Int], value: =>Float): Storage = 
		val newStorage = Array.fill(shape.product)(value)
		new ArrayStorage(newStorage, shape)
	def ones(shape: Seq[Int]): Storage = fill(shape, 1)
	def zeros(shape: Seq[Int]): Storage = fill(shape, 0)
	def rand(shape: Seq[Int]): Storage = 
		val r = Random()
		ArrayStorage.fill(shape, r.nextFloat())
