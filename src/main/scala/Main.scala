package scuda

object Main {
  def main(args: Array[String]): Unit = {
    val tensor1 = Tensor(Seq(1, 2, 3, 4), Seq(2, 2))
    val tensor2 = Tensor(Seq(5, 6, 7, 8), Seq(2, 2))
    println(Seq(2,2)==Seq(2,2))
    // val result = tensor1 + tensor2
    // println(result.storage.sameElements(Array(6, 8, 10, 12)))
  }
}
