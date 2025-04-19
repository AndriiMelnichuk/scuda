import org.scalatest.funsuite.AnyFunSuite
import scuda.tensor.Tensor
import scuda.tensor.storage.Storage
import scuda.tensor.cuda.CudaStorage
import scuda.tensor.cpu.ArrayStorage

class StorageConstructorsTest extends AnyFunSuite:
	/* 
	*	CPU/CUDA TEST 
	*/
	implicit val dhost: String = "cpu"

	test("Storage with shape = seq() or seq(0) throw exception"){
		val s = Seq()
		assertThrows[IllegalArgumentException]{
			val a = Storage(s, s)
		}
	}

	test("Storage (non empty) with shape = seq() or seq(0) throw exception"){
		val s = Seq()
		assertThrows[IllegalArgumentException]{
			val a = Storage(Seq(1), s)
		}
	}

	test("the size of the data must correspond to shape, else exception"){
		val s = 0 until 20 map (_.toFloat)
		val shape = Seq(4, 4)
		assertThrows[IllegalArgumentException]{
			val a = Storage(s, shape)
		}
	}

	test("if shape contains dim <= 0 then throw exception"){
		val s = 0 until 20 map (_.toFloat)
		val shape = Seq(-4, -5)
		assertThrows[IllegalArgumentException]{
			val a = Storage(s, shape)
		}
	}