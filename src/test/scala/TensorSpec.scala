package scuda

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TensorSpec extends AnyFlatSpec with Matchers {

  // init tensor testing
  "Tensor" should "create correctly with CPU storage" in {
    val data = Seq(1.0f, 2.0f, 3.0f)
    val tensor = Tensor(data, "cpu")
    tensor.storage shouldBe a[ArrayStorage]
    tensor.storage.shape should equal(Seq(3))
  }

  it should "create correctly with CUDA storage" in {
    val data = Seq(1.0f, 2.0f, 3.0f)
    val tensor = Tensor(data, "cuda")
    tensor.storage shouldBe a[CudaStorage]
    tensor.storage.shape should equal(Seq(3))
  }

  it should "throw exception for unknown storage type" in {
    val data = Seq(1.0f, 2.0f, 3.0f)
    an [Exception] should be thrownBy {
      val a = Tensor(data, "unknown")
      a.toString()
    }
  }

  it should "create with specific shape for CPU" in {
    val data = Seq(1.0f, 2.0f, 3.0f, 4.0f)
    val shape = Seq(2, 2)
    val tensor = Tensor(data, shape, "cpu")
    tensor.storage shouldBe a[ArrayStorage]
    tensor.storage.shape should equal(shape)
  }

  it should "create with specific shape for CUDA" in {
    val data = Seq(1.0f, 2.0f, 3.0f, 4.0f)
    val shape = Seq(2, 2)
    val tensor = Tensor(data, shape, "cuda")
    tensor.storage shouldBe a[CudaStorage]
    tensor.storage.shape should equal(shape)
  }

  it should "throw exception for invalid shape" in {
    val data = Seq(1.0f, 2.0f, 3.0f)
    val shape = Seq(2, 2)
    an [Exception] should be thrownBy {
      val a = Tensor(data, shape, "cpu")
      a.toString()
    }
  }

  it should "throw exception for empty shape" in {
    val data = Seq(1.0f, 2.0f, 3.0f)
    val shape = Seq.empty[Int]
    an [Exception] should be thrownBy {
      val a = Tensor(data, shape, "cpu")
      a.toString()
    }
  }

  // toCpu and toCuda testing
  "Tensor" should "convert to CPU" in {
    val t = Tensor(Seq(1.0f, 2.0f, 3.0f), "cuda")
    val cpu = t.toCpu()
    cpu.storage shouldBe a[ArrayStorage]
  }

  it should "convert to CUDA" in {
    val t = Tensor(Seq(1.0f, 2.0f, 3.0f), "cpu")
    val cuda = t.toCuda()
    cuda.storage shouldBe a[CudaStorage]
  }

  it should "convert to CPU from CUDA and from CPU to CUDA without data change" in {
    val t = Tensor(Seq(1.0f, 2.0f, 3.0f), "cuda")
    val cpu = t.toCpu()
    cpu.storage.toString should equal(t.storage.toString)
    val cuda = cpu.toCuda().toCpu()
    cuda.storage.toString should equal(cpu.storage.toString)
  }

  // operation testing
  // cpu
  "ArrayStorage" should "perform element-wise addition" in {
    val t1 = Tensor(Seq(1.0f, 2.0f, 3.0f))
    val t2 = Tensor(Seq(4.0f, 5.0f, 6.0f))
    val result = t1 + t2
    result.toString should include("5.0")
    result.toString should include("7.0")
    result.toString should include("9.0")
  }

  it should "perform element-wise subtraction" in {
    val t1 = Tensor(Seq(4.0f, 5.0f, 6.0f))
    val t2 = Tensor(Seq(1.0f, 2.0f, 3.0f))
    val result = t1 - t2
    result.toString should include("3.0")
    result.toString should include("3.0")
    result.toString should include("3.0")
  }

  it should "perform element-wise multiplication" in {
    val t1 = Tensor(Seq(1.0f, 2.0f, 3.0f))
    val t2 = Tensor(Seq(4.0f, 5.0f, 6.0f))
    val result = t1 * t2
    result.toString should include("4.0")
    result.toString should include("10.0")
    result.toString should include("18.0")
  }

  it should "perform element-wise division" in {
    val t1 = Tensor(Seq(4.0f, 10.0f, 18.0f))
    val t2 = Tensor(Seq(2.0f, 5.0f, 6.0f))
    val result = t1 / t2
    result.toString should include("2.0")
    result.toString should include("2.0")
    result.toString should include("3.0")
  }

  it should "perform scalar multiplication" in {
    val t = Tensor(Seq(1.0f, 2.0f, 3.0f))
    val result = t * 2.0f
    result.toString should include("2.0")
    result.toString should include("4.0")
    result.toString should include("6.0")
  }

  it should "perform scalar division" in {
    val t = Tensor(Seq(2.0f, 4.0f, 6.0f))
    val result = t / 2.0f
    result.toString should include("1.0")
    result.toString should include("2.0")
    result.toString should include("3.0")
  }

  it should "perform scalar addition" in {
    val t = Tensor(Seq(1.0f, 2.0f, 3.0f))
    val result = t + 2.0f
    result.toString should include("3.0")
    result.toString should include("4.0")
    result.toString should include("5.0")
  }

  it should "perform scalar subtraction" in {
    val t = Tensor(Seq(3.0f, 4.0f, 5.0f))
    val result = t - 2.0f
    result.toString should include("1.0")
    result.toString should include("2.0")
    result.toString should include("3.0")
  }

  it should "perform matrix multiplication" in {
    val t1 = Tensor(Seq(1.0f, 2.0f, 3.0f, 4.0f), Seq(2, 2))
    val t2 = Tensor(Seq(5.0f, 6.0f, 7.0f, 8.0f), Seq(2, 2))
    val result = t1 ** t2
    result.toString should include("19.0")
    result.toString should include("22.0")
    result.toString should include("43.0")
    result.toString should include("50.0")
  }

  // cuda
  "CudaStorage" should "perform element-wise addition" in {
    val t1 = Tensor(Seq(1.0f, 2.0f, 3.0f), "cuda")
    val t2 = Tensor(Seq(4.0f, 5.0f, 6.0f), "cuda")
    val result = t1 + t2
    result.toString should include("5.0")
    result.toString should include("7.0")
    result.toString should include("9.0")
  }

  it should "perform element-wise subtraction" in {
    val t1 = Tensor(Seq(4.0f, 5.0f, 6.0f), "cuda")
    val t2 = Tensor(Seq(1.0f, 2.0f, 3.0f), "cuda")
    val result = t1 - t2
    result.toString should include("3.0")
    result.toString should include("3.0")
    result.toString should include("3.0")
  }

  it should "perform element-wise multiplication" in {
    val t1 = Tensor(Seq(1.0f, 2.0f, 3.0f), "cuda")
    val t2 = Tensor(Seq(4.0f, 5.0f, 6.0f), "cuda")
    val result = t1 * t2
    result.toString should include("4.0")
    result.toString should include("10.0")
    result.toString should include("18.0")
  }

  it should "perform element-wise division" in {
    val t1 = Tensor(Seq(4.0f, 10.0f, 18.0f), "cuda")
    val t2 = Tensor(Seq(2.0f, 5.0f, 6.0f), "cuda")
    val result = t1 / t2
    result.toString should include("2.0")
    result.toString should include("2.0")
    result.toString should include("3.0")
  }

  it should "perform scalar multiplication" in {
    val t = Tensor(Seq(1.0f, 2.0f, 3.0f), "cuda")
    val result = t * 2.0f
    result.toString should include("2.0")
    result.toString should include("4.0")
    result.toString should include("6.0")
  }

  it should "perform scalar division" in {
    val t = Tensor(Seq(2.0f, 4.0f, 6.0f), "cuda")
    val result = t / 2.0f
    result.toString should include("1.0")
    result.toString should include("2.0")
    result.toString should include("3.0")
  }

  it should "perform scalar addition" in {
    val t = Tensor(Seq(1.0f, 2.0f, 3.0f), "cuda")
    val result = t + 2.0f
    result.toString should include("3.0")
    result.toString should include("4.0")
    result.toString should include("5.0")
  }

  it should "perform scalar subtraction" in {
    val t = Tensor(Seq(3.0f, 4.0f, 5.0f), "cuda")
    val result = t - 2.0f
    result.toString should include("1.0")
    result.toString should include("2.0")
    result.toString should include("3.0")
  }

  it should "perform matrix multiplication" in {
    val t1 = Tensor(Seq[Float](1, 8, 7, 9), Seq(2, 2), "cuda")
    val t2 = Tensor(Seq[Float](5, 1, 6, 9), Seq(2, 2), "cuda")
    val result = t1 ** t2
    result.toString should include("53.0")
    result.toString should include("73.0")
    result.toString should include("89.0")
    result.toString should include("88.0")
  }


  it should "throw exception when operating with different storage types" in {
    val t1 = Tensor(Seq(1.0f, 2.0f, 3.0f), "cuda")
    val t2 = Tensor(Seq(1.0f, 2.0f, 3.0f), "cpu")
    an [Exception] should be thrownBy {
      val a = t1 + t2
      a.toString()
    }
  }

  // beautifulArrayprint testing
  "beautifulArrayprint" should "format array correctly" in {
    val array = Array(1.0f, 2.0f, 3.0f, 4.0f)
    val shape = Seq(2, 2)
    val result = beautifulArrayprint(array, shape)
    result should equal("[[1.0, 2.0],\n[3.0, 4.0]]")
  }
}