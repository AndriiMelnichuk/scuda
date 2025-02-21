package scuda

import scuda._
import javax.print.DocFlavor.STRING
import scala.collection.View.FlatMap

trait GeneralFunction:
	def forward: Storage

trait Funtion2arg extends GeneralFunction:
	val a: Tensor
	val b: Tensor
	def forward: Storage
	def gradLeft: Storage
	def gradRight: Storage

trait Funtion1arg extends GeneralFunction:
	val a: Tensor
	def forward: Storage
	def grad: Storage


class SumTensor(val a: Tensor, val b: Tensor) extends Funtion2arg:
	def forward: Storage = a.storage + b.storage
	def gradLeft: Storage = 
		val storage = a.storage
		val shape = a.storage.shape
		val r = Seq.fill(shape.product)(1f)
		storage match
			case v: ArrayStorage => ArrayStorage(r.toArray, shape)
			case v: CudaStorage => CudaStorage(r, shape)
			
	def gradRight: Storage = gradLeft
		
class ProductTensor(val a: Tensor, val b: Tensor) extends Funtion2arg:
	def forward: Storage = a.storage * b.storage
	def gradLeft: Storage = b.storage
	def gradRight: Storage = a.storage


class CreateTensor(val array: Array[Float], shape: Seq[Int], val isGrad: Boolean) extends GeneralFunction:
	def forward: Storage = ArrayStorage(array, shape)
	
class SinTensor(val a: Tensor) extends Funtion1arg:
	def grad: Storage = 
		a.storage match
			case v: ArrayStorage => ArrayStorage(v.storage.map(math.cos(_).toFloat), v.shape)

	def forward: Storage = 
		a.storage match
			case v: ArrayStorage => ArrayStorage(v.storage.map(math.sin(_).toFloat), v.shape)
		


// def backowrd(x: Tensor) = 
// 	x match
// 		case v: Function2
	
