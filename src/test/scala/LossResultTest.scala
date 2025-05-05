import org.scalatest.funsuite.AnyFunSuite
import scuda.tensor.Tensor
import scuda.tensor.storage.Storage
import scuda.tensor.ai.*

class LossResTest extends AnyFunSuite:

	val d= 0.001
	def tensorEqual(a: Tensor, b: Tensor): Boolean = 
		val st1 = a.storage.toCpu.storage
		val st2 = b.storage.toCpu.storage
		(0 until st1.length)
		.map(i => math.abs(st1(i) - st2(i)) < d).reduce(_ && _) &&  a.storage.shape == b.storage.shape
    
	def storageEqual(a: Storage, b: Storage): Boolean = 
		val sx = a.toCpu.storage
		val sy = b.toCpu.storage
		(0 until sx.length)
		.map(i => math.abs(sx(i) - sy(i)) < d).reduce(_ && _) &&  a.shape == b.shape

	/* 
	*	CPU/CUDA TEST 
	*/
	implicit val dhost: String = "cpu"

	test("cross entropy loss result correct"){
		val x = Tensor(Seq(1f, 2f, 3f, 10f, 18f, 23f), Seq(2, 3), true)
		val y = Tensor(Seq(2f, 0f), Seq(2, 1))

		val layer = StableSoftmax()
		val pr = layer(x)

		val res = crossEntropyLoss(pr, y)

		val tres = Tensor(Seq(6.707161903381348f), Seq(1))

		assert(tensorEqual(tres, res))
	}

	test("mse loss result corect"){
		val x = Tensor(Seq[Float](1, 2, 3, 4, 5, 6), Seq(2, 3), true)
		val y = Tensor(Seq[Float](7, 9, 8, 1, 3, 5), Seq(2, 3))

		val res = MSE(x, y)
		val tres = Tensor(Seq(20.6666f), Seq(1))
		assert(tensorEqual(res, tres))

	}

