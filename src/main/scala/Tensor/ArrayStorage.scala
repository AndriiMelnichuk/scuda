package scuda.Tensor


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

	def T: Storage = 
		if shape.length != 2 then throw new Exception("not 2d Tensor cant be transponed")
		ArrayStorage(storage, shape.reverse)

