import org.scalatest.funsuite.AnyFunSuite
import scuda.tensor.Tensor
import scuda.tensor.storage.Storage
import scuda.tensor.cuda.CudaStorage
import scuda.tensor.cpu.ArrayStorage

class TensorOperationGradTest extends AnyFunSuite:
	def storageEqual(a: Storage, b: Storage): Boolean = 
		val sx = a.toCpu.storage
		val sy = b.toCpu.storage
		val d = 0.0001
		(0 until sx.length)
		.map(i => math.abs(sx(i) - sy(i)) < d).reduce(_ && _) &&  a.shape == b.shape
 
	/* 
	*	CPU/CUDA TEST 
	*/
	implicit val dhost: String = "cpu"

	test("x + y: gradient for x, y must be chainGrad, else zeros"){
		val s1 = 0 until 9 map (_.toFloat)
		val s2 = s1.reverse map (_ + 10)
		val s3 = s1.reverse map (_ + 40)
		val shape = Seq(3, 3)

		val x = Tensor(s1, shape)
		val y = Tensor(s2, shape)
		val t3 = Tensor(s2, shape)
		val chainGrad = Storage(s3, shape)
    
		val res = x + y
		val g1 = res.origin.backward(x, chainGrad)
		val g2 = res.origin.backward(y, chainGrad)
		val g3 = res.origin.backward(t3, chainGrad)

		assert(storageEqual(g1, chainGrad))
		assert(storageEqual(g2, chainGrad))
		assert(storageEqual(g3, Storage.zeros(t3.storage)))
	}

	test("x - y: gradient(x)=chainGrad, gradient(y)=-chainGrad, else zeros"){
		val s1 = 0 until 9 map (_.toFloat)
		val s2 = s1.reverse map (_ + 10)
		val s3 = s1.reverse map (_ + 40)
		val shape = Seq(3, 3)

		val x = Tensor(s1, shape)
		val y = Tensor(s2, shape)
		val t3 = Tensor(s2, shape)
		val chainGrad = Storage(s3, shape)
    
		val res = x - y
		val g1 = res.origin.backward(x, chainGrad)
		val g2 = res.origin.backward(y, chainGrad)
		val g3 = res.origin.backward(t3, chainGrad)

		assert(storageEqual(g1, chainGrad))
		assert(storageEqual(g2, -chainGrad))
		assert(storageEqual(g3, Storage.zeros(t3.storage)))
	}

	test("x * y: gradient(x)=y*chainGrad, gradient(y)=x*chainGrad, else zeros"){
		val s1 = 0 until 9 map (_.toFloat)
		val s2 = s1.reverse map (_ + 10)
		val s3 = s1.reverse map (_ + 40)
		val shape = Seq(3, 3)

		val x = Tensor(s1, shape)
		val y = Tensor(s2, shape)
		val t3 = Tensor(s2, shape)
		val chainGrad = Storage(s3, shape)
    
		val res = x * y
		val g1 = res.origin.backward(x, chainGrad)
		val g2 = res.origin.backward(y, chainGrad)
		val g3 = res.origin.backward(t3, chainGrad)

		assert(storageEqual(g1, y.storage * chainGrad))
		assert(storageEqual(g2, x.storage * chainGrad))
		assert(storageEqual(g3, Storage.zeros(t3.storage)))
	}

	test("x / y: gradient(x)=chainGrad/y gradient(y)=-x*chainGrad/(y pow 2), else zeros"){
		val s1 = 0 until 9 map (_.toFloat)
		val s2 = s1.reverse map (_ + 10)
		val s3 = s1.reverse map (_ + 40)
		val shape = Seq(3, 3)

		val x = Tensor(s1, shape)
		val y = Tensor(s2, shape)
		val t3 = Tensor(s2, shape)
		val chainGrad = Storage(s3, shape)
    
		val res = x / y
		val g1 = res.origin.backward(x, chainGrad)
		val g2 = res.origin.backward(y, chainGrad)
		val g3 = res.origin.backward(t3, chainGrad)

		assert(storageEqual(g1, chainGrad / y.storage))
		assert(storageEqual(g2, -(x.storage) * chainGrad / (y.storage pow 2)))
		assert(storageEqual(g3, Storage.zeros(t3.storage)))
	}

	test("x ** y: gradient(x)=chainGrad**(y.T) gradient(y)=x.T**chainGrad, else zeros"){
		val s1 = 0 until 9 map (_.toFloat)
		val s2 = s1.reverse map (_ + 10)
		val s3 = s1.reverse map (_ + 40)
		val shape = Seq(3, 3)

		val x = Tensor(s1, shape)
		val y = Tensor(s2, shape)
		val t3 = Tensor(s2, shape)
		val chainGrad = Storage(s3, shape)
    
		val res = x ** y
		val g1 = res.origin.backward(x, chainGrad)
		val g2 = res.origin.backward(y, chainGrad)
		val g3 = res.origin.backward(t3, chainGrad)

		assert(storageEqual(g1, chainGrad ** y.storage.T))
		assert(storageEqual(g2, x.storage.T ** chainGrad))
		assert(storageEqual(g3, Storage.zeros(t3.storage)))
	}

	test("x + alpha: gradient for x, must be chainGrad, else zeros"){
		val s1 = 0 until 9 map (_.toFloat)
		val s2 = s1.reverse map (_ + 10)
		val s3 = s1.reverse map (_ + 40)
		val shape = Seq(3, 3)

		val x = Tensor(s1, shape)
		val y = 42f
		val t3 = Tensor(s2, shape)
		val chainGrad = Storage(s3, shape)
    
		val res = x + y
		val g1 = res.origin.backward(x, chainGrad)
		val g3 = res.origin.backward(t3, chainGrad)

		assert(storageEqual(g1, chainGrad))
		assert(storageEqual(g3, Storage.zeros(t3.storage)))
	}

	test("x - alpha: gradient(x)=chainGrad, else zeros"){
		val s1 = 0 until 9 map (_.toFloat)
		val s2 = s1.reverse map (_ + 10)
		val s3 = s1.reverse map (_ + 40)
		val shape = Seq(3, 3)

		val x = Tensor(s1, shape)
		val y = 42f
		val t3 = Tensor(s2, shape)
		val chainGrad = Storage(s3, shape)
    
		val res = x - y
		val g1 = res.origin.backward(x, chainGrad)
		val g3 = res.origin.backward(t3, chainGrad)

		assert(storageEqual(g1, chainGrad))
		assert(storageEqual(g3, Storage.zeros(t3.storage)))
	}

	test("x * alpha: gradient(x)=alpha*chainGrad, else zeros"){
		val s1 = 0 until 9 map (_.toFloat)
		val s2 = s1.reverse map (_ + 10)
		val s3 = s1.reverse map (_ + 40)
		val shape = Seq(3, 3)

		val x = Tensor(s1, shape)
		val y = 42f
		val t3 = Tensor(s2, shape)
		val chainGrad = Storage(s3, shape)
    
		val res = x * y
		val g1 = res.origin.backward(x, chainGrad)
		val g3 = res.origin.backward(t3, chainGrad)

		assert(storageEqual(g1, chainGrad * y))
		assert(storageEqual(g3, Storage.zeros(t3.storage)))
	}

	test("x / alpha: gradient(x)=chainGrad/y, else zeros"){
		val s1 = 0 until 9 map (_.toFloat)
		val s2 = s1.reverse map (_ + 10)
		val s3 = s1.reverse map (_ + 40)
		val shape = Seq(3, 3)

		val x = Tensor(s1, shape)
		val y = 42f
		val t3 = Tensor(s2, shape)
		val chainGrad = Storage(s3, shape)
    
		val res = x / y
		val g1 = res.origin.backward(x, chainGrad)
		val g3 = res.origin.backward(t3, chainGrad)

		assert(storageEqual(g1, chainGrad / y))
		assert(storageEqual(g3, Storage.zeros(t3.storage)))
	}

	test("-x: gradient(x)=-chainGrad, else zeros"){
		val s1 = 0 until 9 map (_.toFloat)
		val s2 = s1.reverse map (_ + 10)
		val s3 = s1.reverse map (_ + 40)
		val shape = Seq(3, 3)

		val x = Tensor(s1, shape)
		val t3 = Tensor(s2, shape)
		val chainGrad = Storage(s3, shape)
    
		val res = -x 
		val g1 = res.origin.backward(x, chainGrad)
		val g3 = res.origin.backward(t3, chainGrad)

		assert(storageEqual(g1, -chainGrad))
		assert(storageEqual(g3, Storage.zeros(t3.storage)))
	}

	test("x.toCpu, when x on cpu: gradient(x)=chainGrada else zeros"){
		val s1 = 0 until 9 map (_.toFloat)
		val s2 = s1.reverse map (_ + 10)
		val s3 = s1.reverse map (_ + 40)
		val shape = Seq(3, 3)

		val x = Tensor(s1, shape)(using "cpu")
		val t3 = Tensor(s2, shape)
		val chainGrad = ArrayStorage(s3, shape)
    
		val res = x.toCpu 
		val g1 = res.origin.backward(x, chainGrad)
		val g3 = res.origin.backward(t3, chainGrad)

		val isCpu = g1 match
			case _: ArrayStorage => true
			case _ => false
		assert(isCpu)
		assert(storageEqual(g1, chainGrad))
		assert(storageEqual(g3, Storage.zeros(t3.storage)))
	}

	test("x.toCpu, when x on cuda: gradient(x)=chainGrada.toCuda else zeros"){
		val s1 = 0 until 9 map (_.toFloat)
		val s2 = s1.reverse map (_ + 10)
		val s3 = s1.reverse map (_ + 40)
		val shape = Seq(3, 3)

		val x = Tensor(s1, shape)(using "cuda")
		val t3 = Tensor(s2, shape)
		val chainGrad = ArrayStorage(s3, shape)
    
		val res = x.toCpu 
		val g1 = res.origin.backward(x, chainGrad)
		val g3 = res.origin.backward(t3, chainGrad)

		val isCuda = g1 match
			case _: CudaStorage => true
			case _ => false
		assert(isCuda)
		assert(storageEqual(g1, chainGrad))
		assert(storageEqual(g3, Storage.zeros(t3.storage)))
	}

	test("x.toCuda, when x on cpu: gradient(x)=chainGrada.toCpu else zeros"){
		val s1 = 0 until 9 map (_.toFloat)
		val s2 = s1.reverse map (_ + 10)
		val s3 = s1.reverse map (_ + 40)
		val shape = Seq(3, 3)

		val x = Tensor(s1, shape)(using "cpu")
		val t3 = Tensor(s2, shape)
		val chainGrad = CudaStorage(s3, shape)
    
		val res = x.toCuda 
		val g1 = res.origin.backward(x, chainGrad)
		val g3 = res.origin.backward(t3, chainGrad)

		val isCpu = g1 match
			case _: ArrayStorage => true
			case _ => false
		assert(isCpu)
		assert(storageEqual(g1, chainGrad))
		assert(storageEqual(g3, Storage.zeros(t3.storage)))
	}

	test("x.toCuda, when x on cuda: gradient(x)=chainGrad else zeros"){
		val s1 = 0 until 9 map (_.toFloat)
		val s2 = s1.reverse map (_ + 10)
		val s3 = s1.reverse map (_ + 40)
		val shape = Seq(3, 3)

		val x = Tensor(s1, shape)(using "cuda")
		val t3 = Tensor(s2, shape)
		val chainGrad = CudaStorage(s3, shape)
    
		val res = x.toCpu 
		val g1 = res.origin.backward(x, chainGrad)
		val g3 = res.origin.backward(t3, chainGrad)

		val isCuda = g1 match
			case _: CudaStorage => true
			case _ => false
		assert(isCuda)
		assert(storageEqual(g1, chainGrad))
		assert(storageEqual(g3, Storage.zeros(t3.storage)))
	}

	test("x pow n: gradient(x) = n * x^(n-1) * chainGrad, else zeros"){
		val s1 = 0 until 9 map (_.toFloat)
		val s2 = s1.reverse map (_ + 10)
		val s3 = s1.reverse map (_ + 40)
		val shape = Seq(3, 3)

		val x = Tensor(s1, shape)
		val n = 8f
		val t3 = Tensor(s2, shape)
		val chainGrad = Storage(s3, shape)
    
		val res = x pow n
		val g1 = res.origin.backward(x, chainGrad)
		val g3 = res.origin.backward(t3, chainGrad)

		val st = x.storage
		assert(storageEqual(g1, (st pow (n - 1)) * n * chainGrad))
		assert(storageEqual(g3, Storage.zeros(t3.storage)))
	}

	test("x.sum: gradient(x)=fill(shape)(chainGrad), else zeros"){
		val s1 = 0 until 9 map (_.toFloat)
		val s2 = s1.reverse map (_ + 10)
		val s3 = Seq(20f)
		val shape = Seq(3, 3)

		val x = Tensor(s1, shape)
		val chainGrad = Storage(s3, Seq(1))
    
		val res = x.sum 
		val g1 = res.origin.backward(x, Storage.fill(Seq(1), 20f))

		assert(storageEqual(g1, Storage.fill(Seq(3, 3), 20f)))
	}

	test("x.T : gradient(x)=chainGrad.T, else zeros"){
		val s1 = 0 until 9 map (_.toFloat)
		val s2 = s1.reverse map (_ + 10)
		val s3 = s1.reverse map (_ + 40)
		val shape = Seq(3, 3)

		val x = Tensor(s1, shape)
		val t3 = Tensor(s2, shape)
		val chainGrad = Storage(s3, shape)
    
		val res = x.T 
		val g1 = res.origin.backward(x, chainGrad)
		val g3 = res.origin.backward(t3, chainGrad)

		assert(storageEqual(g1, chainGrad.T))
		assert(storageEqual(g3, Storage.zeros(t3.storage)))
	}


	test("res.histDrop : gradient(Any) = zeros"){
		val s1 = 0 until 9 map (_.toFloat)
		val s2 = s1.reverse map (_ + 10)
		val s3 = s1.reverse map (_ + 40)
		val shape = Seq(3, 3)

		val x = Tensor(s1, shape)
		val y = Tensor(s2, shape)
		val t3 = Tensor(s2, shape)
		val chainGrad = Storage(s3, shape)
    
		val res = (x + y).historyDrop(true)
		val g1 = res.origin.backward(x, chainGrad)
		val g2 = res.origin.backward(y, chainGrad)
		val g3 = res.origin.backward(t3, chainGrad)

		assert(storageEqual(g1, Storage.zeros(x.storage)))
		assert(storageEqual(g2, Storage.zeros(y.storage)))
		assert(storageEqual(g3, Storage.zeros(t3.storage)))
	}
	

