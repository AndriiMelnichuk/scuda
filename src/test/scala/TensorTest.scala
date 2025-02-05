import org.scalatest.funsuite.AnyFunSuite
import scala.reflect.ClassTag
import scuda.Tensor

class TensorTest extends AnyFunSuite {

  test("Tensor creation with correct data and shape") {
    val tensor = Tensor(Seq(1, 2, 3, 4), Seq(2, 2))
    assert(tensor.storage.sameElements(Array(1, 2, 3, 4)))
    assert(tensor.shape == List(2, 2))
    assert(tensor.device == "cpu")
  }

  test("Tensor creation with different types") {
    val tensorFloat = Tensor(Seq(1.0f, 2.0f, 3.0f, 4.0f), Seq(2, 2))
    assert(tensorFloat.storage.sameElements(Array(1.0f, 2.0f, 3.0f, 4.0f)))

    val tensorDouble = Tensor(Seq(1.0, 2.0, 3.0, 4.0), Seq(2, 2))
    assert(tensorDouble.storage.sameElements(Array(1.0, 2.0, 3.0, 4.0)))
  }


  test("Tensor creation with incorrect data length") {
    assertThrows[ArithmeticException] {
      Tensor(Seq(1, 2, 3), Seq(2, 2))
    }
  }

  test("Tensor creation with unknown device") {
    assertThrows[IllegalArgumentException] {
      Tensor(Seq(1, 2, 3, 4), Seq(2, 2), "unknown")
    }
  }

  test("Tensor toString") {
    val tensor = Tensor(Seq(1, 2, 3, 4), Seq(2, 2))
    assert(tensor.toString == "[[1, 2],\n[3, 4]]")

    val tensor3D = Tensor(Seq(1,2,3,4,5,6,7,8), Seq(2,2,2))
    assert(tensor3D.toString == "[[[1, 2],\n[3, 4]],\n\n[[5, 6],\n[7, 8]]]")

    val emptyTensor = Tensor(Seq.empty[Int], Seq.empty[Int])
    assert(emptyTensor.toString == "[]")
  }

    test("Tensor element-wise addition") {
    val tensor1 = Tensor(Seq(1, 2, 3, 4), Seq(2, 2))
    val tensor2 = Tensor(Seq(5, 6, 7, 8), Seq(2, 2))
    val result = tensor1 + tensor2
    assert(result.storage.sameElements(Array(6, 8, 10, 12)))
  }

  test("Tensor element-wise subtraction") {
    val tensor1 = Tensor(Seq(1, 2, 3, 4), Seq(2, 2))
    val tensor2 = Tensor(Seq(5, 6, 7, 8), Seq(2, 2))
    val result = tensor1 - tensor2
    assert(result.storage.sameElements(Array(-4, -4, -4, -4)))
  }

  test("Tensor element-wise multiplication") {
    val tensor1 = Tensor(Seq(1, 2, 3, 4), Seq(2, 2))
    val tensor2 = Tensor(Seq(5, 6, 7, 8), Seq(2, 2))
    val result = tensor1 * tensor2
    assert(result.storage.sameElements(Array(5, 12, 21, 32)))
  }

//   test("Tensor element-wise operations with different types") {
//     val tensorInt = Tensor(Seq(1, 2, 3, 4), Seq(2, 2))
//     val tensorFloat = Tensor(Seq(5.0f, 6.0f, 7.0f, 8.0f), Seq(2, 2))

//     val resultAdd = tensorInt + tensorFloat
//     assert(resultAdd.storage.sameElements(Array(6.0f, 8.0f, 10.0f, 12.0f)))

//     val resultSub = tensorInt - tensorFloat
//     assert(resultSub.storage.sameElements(Array(-4.0f, -4.0f, -4.0f, -4.0f)))

//     val resultMul = tensorInt * tensorFloat
//     assert(resultMul.storage.sameElements(Array(5.0f, 12.0f, 21.0f, 32.0f)))

//   }

  test("Tensor element-wise operation with incompatible shapes") {
    val tensor1 = Tensor(Seq(1, 2, 3, 4), Seq(2, 2))
    val tensor2 = Tensor(Seq(5, 6, 7, 8, 9, 10), Seq(2, 3))
    assertThrows[Exception] {
      tensor1 + tensor2
    }
  }

  test("Tensor element-wise operation with incompatible devices") {
    val tensor1 = Tensor(Seq(1, 2, 3, 4), Seq(2, 2), "cpu")
    val tensor2 = Tensor(Seq(5, 6, 7, 8), Seq(2, 2), "cuda")
    assertThrows[Exception] {
      tensor1 + tensor2
    }
  }

}