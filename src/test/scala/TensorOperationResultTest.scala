import org.scalatest.funsuite.AnyFunSuite
import scuda.tensor.Tensor
import scuda.tensor.cpu.ArrayStorage
import scuda.tensor.cuda.CudaStorage

class TensorMainOperationTest extends AnyFunSuite:
	def tensorEqual(a: Tensor, b: Tensor): Boolean = 
		val st1 = a.storage.toCpu.storage
		val st2 = b.storage.toCpu.storage
		val d = 0.0001
		(0 until st1.length)
		.map(i => math.abs(st1(i) - st2(i)) < d).reduce(_ && _) &&  a.storage.shape == b.storage.shape

	/* 
	*	CPU/CUDA TEST 
	*/
	implicit val dhost: String = "cpu"
	
	test("+ shoud return the sum of two tensors"){
		val st1 = 0 until 20 map (_.toFloat)
		val st2 = st1 map (_ + 100)
		val shape = Seq(1, 20)
		
		val a = Tensor(st1, shape)
		val b = Tensor(st2, shape)
		val res = a + b
		
		val tres = Tensor((0 until 20).map(i => st1(i) + st2(i)), shape)
		assert(tensorEqual(res, tres))
	}
	
	test("+ isn't work for tensors which shape isn't equal"){
		val st1 = 0 until 20 map (_.toFloat)
		val st2 = st1 map (_ + 100)
		val shape = Seq(1, 20)
		
		val a = Tensor(st1, shape)
		val b = Tensor(st2, shape.reverse)
		val res = a + b
		
		val tres = Tensor((0 until 20).map(i => st1(i) + st2(i)), shape)
		assertThrows[Exception]{
			val essert = tensorEqual(res, tres)
		}
		
	}

	test("- should return the difference of two tensors") {
		val st1 = 0 until 20 map (_.toFloat)
		val st2 = st1 map (_ + 100)
		val shape = Seq(1, 20)

		val a = Tensor(st1, shape)
		val b = Tensor(st2, shape)
		val res = a - b

		val tres = Tensor((0 until 20).map(i => st1(i) - st2(i)), shape)
		assert(tensorEqual(res, tres))
	}

	test("- isn't work for tensors which shape isn't equal") {
		val st1 = 0 until 20 map (_.toFloat)
		val st2 = st1 map (_ + 100)
		val shape = Seq(1, 20)

		val a = Tensor(st1, shape)
		val b = Tensor(st2, shape.reverse)
		val res = a - b

		val tres = Tensor((0 until 20).map(i => st1(i) - st2(i)), shape)
		assertThrows[Exception] {
			val essert = tensorEqual(res, tres)
		}
	}

	test("* should return the elementwise product of two tensors") {
		val st1 = 0 until 20 map (_.toFloat)
		val st2 = st1 map (_ + 100)
		val shape = Seq(1, 20)

		val a = Tensor(st1, shape)
		val b = Tensor(st2, shape)
		val res = a * b

		val tres = Tensor((0 until 20).map(i => st1(i) * st2(i)), shape)
		assert(tensorEqual(res, tres))
	}

	test("* isn't work for tensors which shape isn't equal") {
		val st1 = 0 until 20 map (_.toFloat)
		val st2 = st1 map (_ + 100)
		val shape = Seq(1, 20)

		val a = Tensor(st1, shape)
		val b = Tensor(st2, shape.reverse)
		val res = a * b

		val tres = Tensor((0 until 20).map(i => st1(i) * st2(i)), shape)
		assertThrows[Exception] {
			val essert = tensorEqual(res, tres)
		}
	}

	test("/ should return the division of two tensors") {
		val st1 = 1 until 21 map (_.toFloat)
		val st2 = st1 map (_ + 100)
		val shape = Seq(1, 20)

		val a = Tensor(st1, shape)
		val b = Tensor(st2, shape)
		val res = a / b

		val tres = Tensor((0 until 20).map(i => st1(i) / st2(i)), shape)
		assert(tensorEqual(res, tres))
	}

	test("/ isn't work for tensors which shape isn't equal") {
		val st1 = 1 until 21 map (_.toFloat) 
		val st2 = st1 map (_ + 100)
		val shape = Seq(1, 20)

		val a = Tensor(st1, shape)
		val b = Tensor(st2, shape.reverse)
		val res = a / b

		val tres = Tensor((0 until 20).map(i => st1(i) / st2(i)), shape)
		assertThrows[Exception] {
			val essert = tensorEqual(res, tres)
		}
	}

	test("** shoud return the product of two tensors"){
		val d1 = 0 until 8 map (_.toFloat)
		val s1 = Seq(2, 4)
		val d2 = (0 until 12).map(_.toFloat).reverse.map(_ + 8)
		val s2 = Seq(4, 3)

		val a = Tensor(d1, s1)
		val b = Tensor(d2, s2)
		val res = a ** b
		
		val tres = Tensor(
			Seq(72f, 66f, 60f, 304f, 282f, 260f),
			Seq(2, 3)
		)

		assert(tensorEqual(res, tres))
	}

	test("** shoud throw exception if dim != 2"){
		val d1 = 0 until 8 map (_.toFloat)
		val s1 = Seq(8)
		val d2 = (0 until 12).map(_.toFloat).reverse.map(_ + 8)
		val s2 = Seq(12)

		val a = Tensor(d1, s1)
		val b = Tensor(d2, s2)
		val res = a ** b
		
		assertThrows[Exception] {
			val essert = res.toString()
		}
	}

	test("** shoud throw exception if dim2 of Tensor1 != dim1 of Tensor2"){
		val d1 = 0 until 8 map (_.toFloat)
		val s1 = Seq(2, 4)
		val d2 = (0 until 12).map(_.toFloat).reverse.map(_ + 8)
		val s2 = Seq(2, 6)

		val a = Tensor(d1, s1)
		val b = Tensor(d2, s2)
		val res = a ** b
		
		assertThrows[Exception] {
			val essert = res.toString()
		}
	}

	test("+ with scalar should add scalar to each element") {
		val st: IndexedSeq[Float] = (0 until 20).map(_.toFloat)
		val shape: Seq[Int] = Seq(1, 20)
		val alpha: Float = 10.5f

		val a = Tensor(st, shape)
		val res = a + alpha
		val expected = Tensor(st.map(_ + alpha), shape)
		assert(tensorEqual(res, expected))
	}

	test("- with scalar should subtract scalar from each element") {
		val st: IndexedSeq[Float] = (0 until 20).map(_.toFloat)
		val shape: Seq[Int] = Seq(1, 20)
		val alpha: Float = 10.5f

		val a = Tensor(st, shape)
		val res = a - alpha
		val expected = Tensor(st.map(_ - alpha), shape)
		assert(tensorEqual(res, expected))
	}

	test("* with scalar should multiply each element by scalar") {
		val st: IndexedSeq[Float] = (0 until 20).map(_.toFloat)
		val shape: Seq[Int] = Seq(1, 20)
		val alpha: Float = 10.5f

		val a = Tensor(st, shape)
		val res = a * alpha
		val expected = Tensor(st.map(_ * alpha), shape)
		assert(tensorEqual(res, expected))
	}

	test("/ with scalar should divide each element by scalar") {
		val st: IndexedSeq[Float] = (0 until 20).map(_.toFloat)
		val shape: Seq[Int] = Seq(1, 20)
		val alpha: Float = 10.5f

		val a = Tensor(st, shape)
		val res = a / alpha
		val expected = Tensor(st.map(_ / alpha), shape)
		assert(tensorEqual(res, expected))
	}

	test("T shoud transpose the tensor"){
		val d = 0 until 8 map (_.toFloat)
		val s = Seq(4, 2)
		val t = Tensor(d, s)

		val res = t.T
		val tres = Tensor(
			Seq(0f, 2f, 4f, 6f,
					1f, 3f, 5f, 7f),
			Seq(2, 4))

		assert(tensorEqual(res, tres))
	}

	test("T shoud only if dim == 2"){
		val d = 0 until 8 map (_.toFloat)
		
		val s1 = Seq(8)
		val t1 = Tensor(d, s1)

		val s2 = Seq(2, 2, 2)
		val t2 = Tensor(d, s2)

		assertThrows[Exception] {
			val essert = t1.T.toString()
		}

		assertThrows[Exception] {
			val essert = t2.T.toString()
		}
	}

	test("toCpu must convert Storage to ArrayStorage"){
		val d = 0 until 8 map (_.toFloat)
		val s = Seq(4, 2)
		
		val t1 = Tensor(d, s)(using "cpu")
		val isCpuToCpu = t1.toCpu.storage match
			case _: ArrayStorage => true
			case _ => false
		assert(isCpuToCpu)

		val t2 = Tensor(d, s)(using "cuda")
		val isCudaToCpu = t2.toCpu.storage match
			case _: ArrayStorage => true
			case _ => false
		assert(isCpuToCpu)
	}

	test("toCuda must convert Storage to CudaStorage"){
		val d = 0 until 8 map (_.toFloat)
		val s = Seq(4, 2)
		
		val t1 = Tensor(d, s)(using "cpu")
		val isCpuToCuda = t1.toCuda.storage match
			case _: CudaStorage => true
			case _ => false
		assert(isCpuToCuda)

		val t2 = Tensor(d, s)(using "cuda")
		val isCudaToCuda = t2.toCuda.storage match
			case _: CudaStorage => true
			case _ => false
		assert(isCudaToCuda)
	}

	test("sum must sum all elements at tensor"){
		val d = 0 until 8 map (_.toFloat + 21)
		val s = Seq(4, 2)
		
		val t = Tensor(d, s)
		val res = t.sum

		val tRes = Tensor(Seq(d.sum), Seq(1))
		assert(res.item == d.sum)
	}

	test("sum must return tensor with shape == Seq(1)"){
		val d = 0 until 8 map (_.toFloat + 21)
		val s = Seq(4, 2)
		
		val t = Tensor(d, s)
		val res = t.sum

		assert(res.storage.shape == Seq(1))
	}

	test("item must return first element in storage"){
		val d = Seq(42f)
		val s = Seq(1)
		val t = Tensor(d, s)
		
		assert(t.item == 42f)
	}

	test("item must throw exception if shape != Seq(1)"){
		val d = 0 until 8 map (_.toFloat + 21)
		val s = Seq(4, 2)
		
		val t = Tensor(d, s)

		assertThrows[Exception]{
			val res = t.item
		}
	}

	test("unary_- must be equal Tensor * -1"){
		val st: IndexedSeq[Float] = (0 until 20).map(_.toFloat)
		val shape: Seq[Int] = Seq(1, 20)

		val a = Tensor(st, shape)
		val res = -a
		val expected = Tensor(st.map(-_), shape)
		assert(tensorEqual(res, expected))
	}

	test("pow must raise all elements to degree n"){
		val st: IndexedSeq[Float] = (0 until 20).map(_.toFloat)
		val shape: Seq[Int] = Seq(1, 20)

		val a = Tensor(st, shape)
		val res = a pow 2
		val expected = Tensor(st.map(math.pow(_, 2).toFloat), shape)
		assert(tensorEqual(res, expected))
	}

