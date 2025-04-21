import org.scalatest.funsuite.AnyFunSuite
import scuda.tensor.Tensor
import scuda.tensor.storage.Storage
import scuda.tensor.ai.*

class LayerGradTest extends AnyFunSuite:
	val d= 0.001
	def tensorEqual(a: Tensor, b: Tensor): Boolean = 
		val st1 = a.storage.toCpu().storage
		val st2 = b.storage.toCpu().storage
		(0 until st1.length)
		.map(i => math.abs(st1(i) - st2(i)) < d).reduce(_ && _)
    
	def storageEqual(a: Storage, b: Storage): Boolean = 
		val sx = a.toCpu().storage
		val sy = b.toCpu().storage
		(0 until sx.length)
		.map(i => math.abs(sx(i) - sy(i)) < d).reduce(_ && _)

	/* 
	*	CPU/CUDA TEST 
	*/
	implicit val dhost: String = "cpu"


	test("Forward layer computation correct"){
		// Forward init
		val w_d = Seq(-0.4100f, -0.0681f,  0.0581f,  0.0486f, 
									-0.1011f, -0.2201f,  0.0224f,  0.3765f)
		val w_s = Seq(2, 4)
		val w = Tensor(w_d, w_s, true)

		val b_d = Seq(0.1471f, 0.3450f)
		val b_s = Seq(2, 1)
		val b = Tensor(b_d, b_s, true)

		val x_d = Seq(1.4393f, 1.1543f, 2.2369f, 0.5015f)
		val x_s = Seq(1, 4)
		val x = Tensor(x_d, x_s)

		val y_d = Seq(-0.1f,  0.2f)
		val y_s = Seq(1,2)
		val y = Tensor(y_d, y_s)

		val layer = new ForwardLayer(w, b)
		val pr = layer(x)

		// loss
		val loss = ((y - pr) pow 2).sum / 2
		
		val gr = gradientSearch(loss)
		
		val w_g = gr(w)
		val w_tg = Storage(
			Seq(-0.3847f, -0.3085f, -0.5979f, -0.1340f,
				  -0.0225f, -0.0181f, -0.0350f, -0.0079f),
			Seq(2, 4)
		)

		val b_g = gr(b)
		val b_tg = Storage(
			Seq(-0.2673f, -0.0157f),
			Seq(2, 1)
		)

		assert(storageEqual(w_g, w_tg))
		assert(storageEqual(b_g, b_tg))

	}

	test("ReLU layer computation correct"){
		val x = Tensor(Seq[Float](1, -5, 0, 9, 8, -3.5), Seq(3,2), true)
		val layer = ReLU()
		val res = layer(x)

		val loss = res.sum * 2

		val grad = gradientSearch(loss)
		val x_g = grad(x)
		val x_tg = Storage(
			Seq(2f, 0f, 0f, 2f, 2f, 0f),
			Seq(3, 2)
		)


		assert(storageEqual(x_tg, x_g))
	}

	test("Sequetial layer computation correct"){
		// ===== FC1 =====
		val fc1_w_d = Seq(
			0.4630f, -0.3816f,  0.0642f, -0.4721f,
			0.1819f, -0.4430f, -0.4186f, -0.3929f,
			0.0884f, -0.4871f, -0.3401f, -0.0609f,
			0.0405f, -0.3529f, -0.3751f,  0.3466f,
		-0.3144f, -0.4563f,  0.4421f, -0.4925f,
			0.1590f, -0.1329f, -0.4013f,  0.0067f,
			0.2296f,  0.1829f, -0.1371f,  0.2442f,
		-0.1490f,  0.3087f,  0.2192f, -0.3909f
		)
		val fc1_w_s = Seq(8, 4)
		val fc1_w = Tensor(fc1_w_d, fc1_w_s, isGrad = true)

		val fc1_b_d = Seq(
			0.3519f, -0.1588f, 0.1174f, -0.2038f, 
			0.0750f, -0.1427f, 0.1398f, -0.3562f
		)
		val fc1_b_s = Seq(8,1)
		val fc1_b = Tensor(fc1_b_d, fc1_b_s, isGrad = true)

		// ===== FC2 =====
		val fc2_w_d = Seq(
			0.0571f, -0.2067f, 0.0315f, 0.1392f, -0.2732f, 0.1340f, 0.0941f, 0.1912f,
			0.3217f, -0.2784f, 0.0943f, -0.0251f, -0.0417f, 0.1751f, -0.2413f, 0.2668f
		)
		val fc2_w_s = Seq(2, 8)
		val fc2_w = Tensor(fc2_w_d, fc2_w_s, isGrad = true)

		val fc2_b_d = Seq(-0.1544f, 0.2361f)
		val fc2_b_s = Seq(2,1 )
		val fc2_b = Tensor(fc2_b_d, fc2_b_s, isGrad = true)

		// ===== Input Tensor =====
		val x_d = Seq(0.7566f,  1.1043f, -1.2939f, -0.0228f)
		val x_s = Seq(1, 4)
		val x = Tensor(x_d, x_s)

		// ===== Output (Example) =====
		val y_d = Seq(-0.3f,  0.8f)
		val y_s = Seq(1, 2)
		val y = Tensor(y_d, y_s)

		val layer = Sequential(Seq(
			new ForwardLayer(fc1_w, fc1_b),
			new ReLU(),
			new ForwardLayer(fc2_w, fc2_b)
		))

		val pr = layer(x)
		val loss = ((y - pr) pow 2).sum / 2

		val grad = gradientSearch(loss)

		// ===== ∂L/∂fc2_w =====
		val d_fc2_w_d = Seq(
			0.0549f,  0.0106f,  0.0231f,  0.0000f,  0.0000f,  0.0922f,  0.1811f,  0.0000f,
			-0.1260f, -0.0243f, -0.0531f, -0.0000f, -0.0000f, -0.2115f, -0.4154f, -0.0000f
		)
		val d_fc2_w_s = Seq(2, 8)
		val fc2_w_tg = Storage(d_fc2_w_d, d_fc2_w_s)

		// ===== ∂L/∂fc2_b =====
		val d_fc2_b_d = Seq(0.2635f, -0.6043f)
		val d_fc2_b_s = Seq(2, 1)
		val fc2_b_tg = Storage(d_fc2_b_d, d_fc2_b_s)

		// ===== ∂L/∂fc1_w =====
		val d_fc1_w_d = Seq(
			-0.1357f, -0.1981f,  0.2321f,  0.0041f,
			0.0861f,  0.1256f, -0.1472f, -0.0026f,
			-0.0368f, -0.0538f,  0.0630f,  0.0011f,
			0.0000f,  0.0000f, -0.0000f, -0.0000f,
			0.0000f,  0.0000f, -0.0000f, -0.0000f,
			-0.0533f, -0.0779f,  0.0912f,  0.0016f,
			0.1291f,  0.1884f, -0.2208f, -0.0039f,
			0.0000f,  0.0000f, -0.0000f, -0.0000f
		)
		val d_fc1_w_s = Seq(8, 4)
		val fc1_w_tg = Storage(d_fc1_w_d, d_fc1_w_s)

		// ===== ∂L/∂fc1_b =====
		val d_fc1_b_d = Seq(
			-0.1794f,  0.1138f, -0.0487f,  0.0000f,
			0.0000f, -0.0705f,  0.1706f,  0.0000f
		)
		val d_fc1_b_s = Seq(8, 1)
		val fc1_b_tg = Storage(d_fc1_b_d, d_fc1_b_s)

		assert(storageEqual(fc1_w_tg, grad(fc1_w)))
		assert(storageEqual(fc1_b_tg, grad(fc1_b)))
		assert(storageEqual(fc2_w_tg, grad(fc2_w)))
		assert(storageEqual(fc2_b_tg, grad(fc2_b)))
	}

	test("StableSoftmax layer computation correct"){
		val x = Tensor(Seq[Float](1f, 2f, 3f, 8f, 7f, 10f), Seq(2, 3), true)
		val y = Tensor(Seq(0f, 1f, 0f, 1f, 0f, 0.8f), Seq(2, 3))
		val tg = Storage(Seq[Float](-0.0220f,  0.1848f, -0.1628f, 0.0241f, -0.0332f,  0.0091f), Seq(2, 3))

		val layer = StableSoftmax()
		val loss = (layer(x) * y).sum
		val grad = gradientSearch(loss)
		val g = grad(x)
		assert(storageEqual(tg, g))
	}