package scuda.Tensor



// def onesLike(a: Tensor): Storage = 
// 	val storage = a.storage
// 	val shape = a.storage.shape
// 	val r = Seq.fill(shape.product)(1f)
// 	storage match
// 		case v: ArrayStorage => ArrayStorage(r.toArray, shape)
// 		case v: CudaStorage => CudaStorage(r, shape)

class GeneralFunction(
	arguments: => Seq[Tensor], 
	forwardFun: => Storage, 
	backwardFun: => Seq[() => Storage],
	reducerFun: => Seq[Storage => Storage]
):
	def args: Seq[Tensor] = arguments
	def forward: Storage = forwardFun
	def backward: Seq[() => Storage] = backwardFun
	def reducer: Seq[Storage => Storage] = reducerFun

// def TensorPlus(a: Tensor, b: Tensor) =
// 	GeneralFunction(
// 		Seq(a,b),
// 		a.storage + b.storage,
// 		Seq(() => onesLike(a), () => onesLike(a))
// 	)




// class CreateTensor(val array: Array[Float], shape: Seq[Int], val isGrad: Boolean) extends GeneralFunction:
// 	def args: Seq[Tensor] = ???
// 	def backward: Seq[()  => Storage] = ???

// 	def forward: Storage = ArrayStorage(array, shape)
	

// class SinTensor(val a: Tensor) extends GeneralFunction:
// 	def args: Seq[Tensor] = Seq(a)
// 	def backward: Seq[()  => Storage] = Seq(() => grad)

// 	def grad: Storage = 
// 		a.storage match
// 			case v: ArrayStorage => ArrayStorage(v.storage.map(math.cos(_).toFloat), v.shape)

// 	def forward: Storage = 
// 		a.storage match
// 			case v: ArrayStorage => ArrayStorage(v.storage.map(math.sin(_).toFloat), v.shape)
		

// class DiffTensor(val a: Tensor, val b: Tensor) extends GeneralFunction:
// 	def args: Seq[Tensor] = Seq(a, b)

// 	def forward: Storage = a.storage - b.storage

// 	def backward: Seq[() => Storage] = 
// 		Seq(() => onesLike(a), () => onesLike(a) * -1f)

