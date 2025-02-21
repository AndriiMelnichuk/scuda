package scuda

import scuda._
import javax.print.DocFlavor.STRING
import scala.collection.View.FlatMap

trait GeneralFunction:
	def args: Seq[Tensor]
	def forward: Storage
	def backward: Seq[() => Storage]


class SumTensor(val a: Tensor, val b: Tensor) extends GeneralFunction:
	def args: Seq[Tensor] = Seq(a, b)

	def backward: Seq[()  => Storage] = 
		Seq(() => gradOnes, () => gradOnes)

	def forward: Storage = a.storage + b.storage

	def gradOnes: Storage = 
		val storage = a.storage
		val shape = a.storage.shape
		val r = Seq.fill(shape.product)(1f)
		storage match
			case v: ArrayStorage => ArrayStorage(r.toArray, shape)
			case v: CudaStorage => CudaStorage(r, shape)
			

class ProductTensor(val a: Tensor, val b: Tensor) extends GeneralFunction:
	def args: Seq[Tensor] = Seq(a, b)

	def backward: Seq[()  => Storage] = Seq(
		() => b.storage,
		() => a.storage
	)

	def forward: Storage = a.storage * b.storage


class CreateTensor(val array: Array[Float], shape: Seq[Int], val isGrad: Boolean) extends GeneralFunction:
	def args: Seq[Tensor] = ???
	def backward: Seq[()  => Storage] = ???

	def forward: Storage = ArrayStorage(array, shape)
	
class SinTensor(val a: Tensor) extends GeneralFunction:
	def args: Seq[Tensor] = Seq(a)
	def backward: Seq[()  => Storage] = Seq(() => grad)

	def grad: Storage = 
		a.storage match
			case v: ArrayStorage => ArrayStorage(v.storage.map(math.cos(_).toFloat), v.shape)

	def forward: Storage = 
		a.storage match
			case v: ArrayStorage => ArrayStorage(v.storage.map(math.sin(_).toFloat), v.shape)
		


// def backowrd(x: Tensor) = 
// 	x match
// 		case v: Function2
	
