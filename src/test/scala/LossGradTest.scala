import org.scalatest.funsuite.AnyFunSuite
import scuda.tensor.Tensor
import scuda.tensor.storage.Storage
import scuda.tensor.ai.*


class LossGradTest extends AnyFunSuite:

	val d= 0.001
	def tensorEqual(a: Tensor, b: Tensor): Boolean = 
		val st1 = a.storage.toCpu().storage
		val st2 = b.storage.toCpu().storage
		(0 until st1.length)
		.map(i => math.abs(st1(i) - st2(i)) < d).reduce(_ && _) &&  a.storage.shape == b.storage.shape
	
	def storageEqual(a: Storage, b: Storage): Boolean = 
		val sx = a.toCpu().storage
		val sy = b.toCpu().storage
		(0 until sx.length)
		.map(i => math.abs(sx(i) - sy(i)) < d).reduce(_ && _) &&  a.shape == b.shape

	/* 
	*	CPU/CUDA TEST 
	*/
	implicit val dhost: String = "cuda"

	test("cross entropy loss grad correct"){
		val x = Tensor(Seq(1f, 2f, 3f, 10f, 18f, 23f), Seq(2, 3), true)
		val y = Tensor(Seq(2f, 0f), Seq(2, 1))

		val layer = StableSoftmax()
		val pr = layer(x)

		val res = crossEntropyLoss(pr, y)
		val grad = gradientSearch(res)

		val g = grad(x)
		val tg = Storage(Seq[Float](0.0450,  0.1224, -0.1674, -0.5000,  0.0033,  0.4967), Seq(2, 3))
		

		assert(storageEqual(g, tg))
	}

	test("mse loss grad corect"){
		val x = Tensor(Seq[Float](1, 2, 3, 4, 5, 6), Seq(2, 3), true)
		val y = Tensor(Seq[Float](7, 9, 8, 1, 3, 5), Seq(2, 3), true)

		val res = MSE(x, y)
		val grad = gradientSearch(res)
		val x_g = grad(x)
		val y_g = grad(y)

		val x_tg = Storage(Seq[Float](-2.0000, -2.3333, -1.6666, 1.0000, 0.6666, 0.3333), Seq(2, 3))
		val y_tg = Storage(Seq[Float](2.0000, 2.3333, 1.6666, -1.0000, -0.6666, -0.3333), Seq(2, 3))
		assert(storageEqual(x_g, x_tg))
		assert(storageEqual(y_g, y_tg))
	}