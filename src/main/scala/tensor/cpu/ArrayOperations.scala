package scuda.tensor.cpu

import scala.math.{ log, exp}
import scala.collection.parallel.CollectionConverters._
import scuda.tensor.storage.Storage

def relu(x: ArrayStorage) = 
  ArrayStorage(x.storage.map(v => if (v > 0) v else 0f), x.shape)

def reluGrad(x: ArrayStorage, cg: ArrayStorage) =
  val res = ArrayStorage(x.storage.map(x => if x > 0 then 1f else 0f), x.shape)
  res * cg


/**
  * 
  *
  * @param pr - M x N
  * @param target N x 1
  * @return
  */
def crossEntropyLoss(pr: ArrayStorage, target: ArrayStorage): ArrayStorage =
  val m = pr.shape(0)
  val n = pr.shape(1)
  val intTarget = target.storage map (x => x.toInt)
  val res = (0 until m).par
    .map(x => x * n + intTarget(x))
    .map(x => -log(pr.storage(x)).toFloat)
    .toArray
  new ArrayStorage(res, target.shape)

def crossEntropyLossGrad(pr: ArrayStorage, target: ArrayStorage, chainGrad: ArrayStorage): ArrayStorage =
  val m = pr.shape(0)
  val n = pr.shape(1)
  val iTarget = target.storage.map(x => x.toInt)
  val res = for
    i <- 0 until m
    j <- 0 until n
  yield if target.storage(i) == j then chainGrad.storage(i) / pr.storage(i * n + j) else 0
  new ArrayStorage(res.toArray, pr.shape)

def stableSoftmax(x: ArrayStorage): ArrayStorage = 
  val m = x shape 0
  val n = x shape 1

  val st = x.storage
  val max = (st grouped n map (x => x.max)).toList
  val ste = (0 until m * n).par.map(i => exp(st(i) - max(i / n)).toFloat).toArray
  // val ste = st.par.map(x => exp(x).toFloat).toArray
  val sums = (ste grouped n).map(_.sum).toArray

  val res = (0 until m * n ).par.map(i => ste(i) / sums(i / n)).toArray
  ArrayStorage(res, x.shape)
  
def stableSoftmaxGrad(sm: ArrayStorage, cg: ArrayStorage): ArrayStorage =
  // TODO need test
  val m = sm shape 0
  val n = sm shape 1
  def dldx(i: Int, j: Int) =
    (0 until n).par map ( k =>
      if k == j then
        sm.storage(i * n + j) * (1 - sm.storage(i * n + j)) * cg.storage(i * n + k)
      else
        -sm.storage(i * n + j) * sm.storage(i * n + k) * cg.storage(i * n + k)
    ) reduce (_ + _)
  
  val res = for
    i <- (0 until m).par
    j <- 0 until n
  yield dldx(i, j)

  ArrayStorage(res.toArray, sm.shape)

def softmax(x: ArrayStorage): ArrayStorage = 
  val m = x shape 0
  val n = x shape 1

  val st = x.storage
  val ste = st.par.map(x => exp(x).toFloat).toArray
  val sums = (ste grouped n).map(_.sum).toArray

  val res = (0 until m * n ).par.map(i => ste(i) / sums(i / n)).toArray
  ArrayStorage(res, x.shape)

def crossEntropyLogitsLoss(logits: ArrayStorage, target: ArrayStorage) =
  val m = logits shape 0
  val n = logits shape 1
  val sm = softmax(logits)
  
  var res = (0 until m).par
  .map(i => sm.storage(target.storage(i).toInt + i * n))
  .map(x => -log(x).toFloat)
  .sum / m
  
  ArrayStorage(Array(res), Seq.fill(target.shape.length)(1))