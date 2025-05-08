package scuda.tensor.cpu

import scuda.tensor.storage.Storage
import scuda.tensor.cuda.CudaStorage
import scuda.tensor.utils.{ beautifulArrayprint, host2device }
import scala.collection.parallel.CollectionConverters._
import scala.util.Random
import scala.annotation.tailrec

class ArrayStorage(
	val storage: Array[Float], 
	val shape: Seq[Int]
	) extends Storage:
	

	require(storage.nonEmpty, "ArrayStorage: storage array must not be empty")
	require(shape != Seq(), "ArrayStorage: shape must not be empty")
	require(storage.length == shape.product, "ArrayStorage: size of the data must correspond to shape")
	require(shape.map(_ > 0).reduce(_ && _), "ArrayStorage: shape must not contain dim < 0")

	def elementByElementOperation(operation: (Float, Float) => Float)(other: Storage): ArrayStorage = 
		other match
			case other: ArrayStorage => 
				if this.shape != other.shape then 
					if shape.product > other.shape.product then elementByElementOperation(operation)(broadcasting(other, shape))
					else broadcasting(this, other.shape).elementByElementOperation(operation)(other)
				else ArrayStorage(storage.zip(other.storage).map(operation(_,_)), shape)
			case _ => throw new Exception("Operation cannot be performed if devise isn't same.")
	
	override def toString(): String = beautifulArrayprint(storage, shape)

	def +(other: Storage) = elementByElementOperation(_ + _)(other)
	def -(other: Storage) = elementByElementOperation(_ - _)(other)
	def *(other: Storage) = elementByElementOperation(_ * _)(other)
	def /(other: Storage) = elementByElementOperation(_ / _)(other)
	def **(other: Storage): ArrayStorage = 
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

	def +(alpha: Float): ArrayStorage = new ArrayStorage(storage.map(_ + alpha), shape)
	def -(alpha: Float): ArrayStorage = new ArrayStorage(storage.map(_ - alpha), shape)
	def *(alpha: Float): ArrayStorage = new ArrayStorage(storage.map(_ * alpha), shape)
	def /(alpha: Float): ArrayStorage = new ArrayStorage(storage.map(_ / alpha), shape)    
	def pow(n: Float): ArrayStorage =
		new ArrayStorage(storage.par.map(x => math.pow(x.toDouble,n).toFloat).toArray, shape)
	def unary_- : ArrayStorage = new ArrayStorage(storage.map(-_), shape)

	// device change
	def toCpu: ArrayStorage = this
	def toCuda: CudaStorage = new CudaStorage(host2device(storage, shape.product), shape)

	// reduce
	def sum: ArrayStorage = ArrayStorage(Array(storage.par.reduce(_ + _)), Seq(1))
	def item: Float = 
		if shape == Seq(1) then storage(0)
		else throw new Exception("It is impossible to take an element from the storage if shape != Seq(1)")

	def T: ArrayStorage = 
		if shape.length != 2 then throw new Exception("not 2d Tensor cant be transponed")
		val newStorage =
		for
			i <- 0 until shape(1)
			j <- 0 until shape(0)
		yield storage(j * shape(1) + i)

		ArrayStorage(newStorage.toArray, shape.reverse)

	def reshape(seq: Iterable[Int]): ArrayStorage = 
		ArrayStorage(storage, seq.toSeq)

	def apply(args: Seq[Int | Iterable[Int]]): ArrayStorage =
		def selectByAxis(x: ArrayStorage, axis: Int = 0, axisN: Int = 0): ArrayStorage =
			if axisN == -1 then return x
			if axisN > x.shape(axis) - 1 then throw Exception("invalid axis number")

			val nShape = x.shape.patch(axis,Nil,1)
			val s      = x.shape.reverse.scanLeft(1)(_*_).init.reverse(axis)
			val a      = x.shape(axis)

			val resIdx = for 
				n <- 0 until nShape.product / s
				r <- 0 until s
			yield s * (a * n + axisN) + r
			val res = resIdx.map(x.storage)

			ArrayStorage(res, nShape)

		def selectByAxes(x: ArrayStorage, axis: Int, axisN: Seq[Int]): ArrayStorage =
			if axisN.length > x.shape(axis) then throw new Exception("")
			if axisN.map(_ > x.shape(axis) - 1).reduce(_ || _) then throw Exception("")
			
			axisN.map{
				selectByAxis(x, axis, _).reshape(x.shape.updated(axis, 1))
			}
			.reduce((x, y) => x.cat(y, axis))

		@tailrec
		def helper(st: ArrayStorage, args: Seq[Int | Iterable[Int]], ax: Int = 0): ArrayStorage = 
			args match
				case Nil => st
				case (i: Int) +: tail => 
					val res = selectByAxis(st, ax, i)
					val nax = if res.shape.length < st.shape.length then ax else ax + 1
					helper(res, tail, nax)
				case (it: Iterable[Int]) +: tail => 
					val res = selectByAxes(st, ax, it.toSeq)
					val nax = if res.shape.length < st.shape.length then ax else ax + 1
					helper(res, tail, nax)
		if args.length > shape.length then throw new Exception("Too many axes")
		helper(this, args)
		
	def cat(st: Storage, dim: Int = 0): ArrayStorage = 
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
	def fill(shape: Seq[Int], value: =>Float): ArrayStorage = 
		val newStorage = Array.fill(shape.product)(value)
		new ArrayStorage(newStorage, shape)
	def ones(shape: Seq[Int]): ArrayStorage = fill(shape, 1)
	def zeros(shape: Seq[Int]): ArrayStorage = fill(shape, 0)
	def rand(shape: Seq[Int]): ArrayStorage = 
		val r = Random()
		ArrayStorage.fill(shape, r.nextFloat())
