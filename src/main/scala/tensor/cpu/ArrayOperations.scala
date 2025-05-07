package scuda.tensor.cpu

import scala.collection.parallel.CollectionConverters._
import scuda.tensor.storage.Storage

def relu(x: ArrayStorage) = 
	ArrayStorage(x.storage.map(v => if (v > 0) v else 0f), x.shape)

def reluGrad(x: ArrayStorage, cg: ArrayStorage) =
	val res = ArrayStorage(x.storage.map(x => if x > 0 then 1f else 0f), x.shape)
	res * cg

def sigmoid(x: ArrayStorage): ArrayStorage =
	val res = x.storage.par.map(x => 1f / (1f + math.exp(-x).toFloat))
	ArrayStorage(res.toArray, x.shape)

def tanh(x: ArrayStorage): ArrayStorage =
	val res = x.storage.par.map(x => math.tanh(x).toFloat)
	ArrayStorage(res.toArray, x.shape)

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
		.map(x => -math.log(pr.storage(x)).toFloat)
		.toArray
	new ArrayStorage(res, target.shape)

def crossEntropyLossGrad(pr: ArrayStorage, target: ArrayStorage, chainGrad: ArrayStorage): ArrayStorage =
	val m = pr.shape(0)
	val n = pr.shape(1)
	val iTarget = target.storage.map(x => x.toInt)
	val res = for
		i <- 0 until m
		j <- 0 until n
	yield if target.storage(i) == j then -chainGrad.storage(i) / pr.storage(i * n + j) else 0
	new ArrayStorage(res.toArray, pr.shape)

def stableSoftmax(x: ArrayStorage): ArrayStorage = 
	val m = x shape 0
	val n = x shape 1

	val st = x.storage
	val max = (st grouped n map (x => x.max)).toList
	val ste = (0 until m * n).par.map(i => math.exp(st(i) - max(i / n)).toFloat).toArray
	val sums = (ste grouped n).map(_.sum).toArray

	val res = (0 until m * n ).par.map(i => ste(i) / sums(i / n)).toArray
	ArrayStorage(res, x.shape)
	
def stableSoftmaxGrad(sm: ArrayStorage, cg: ArrayStorage): ArrayStorage =
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

def broadcasting(x: ArrayStorage, shape: Seq[Int]): ArrayStorage =
	def shapeCheck(rs: Seq[Int], es: Seq[Int]): Unit = 
		if es.length > rs.length then
			val res = Seq.fill(es.length - rs.length)(1) concat rs
			shapeCheck(res, es)
		else if es.length == rs.length then
			(es zip rs).foreach{ (e, r) =>
				if r > e then
					throw new Exception(s"broadcasting: real shape > expected shape. rs: $rs, es: $es")
				else if r != 1 && r != e then 
					throw new Exception(s"broadcasting: expected shape: $e, real shape: $r")

			}
		else 
			throw new Exception(s"broadcasting: real shape len > expected shape len. rs: $rs, es: $es")
	def recursiveBroadcasting(a: Array[Float], n: Int): ArrayStorage =
		if n == -1                     then ArrayStorage(a, shape)
		else if x.shape(n) == shape(n) then recursiveBroadcasting(a, n - 1)
		else
			val elems2copy = shape.drop(n + 1).product
			val copyCount = shape(n)
			val res = a.grouped(elems2copy).flatMap { group =>
				Array.fill(copyCount)(group).flatten
			}.toArray
			recursiveBroadcasting(res, n - 1)
	shapeCheck(x.shape, shape)
	recursiveBroadcasting(x.storage, shape.length - 1)