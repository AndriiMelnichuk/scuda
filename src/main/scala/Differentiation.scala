package scuda

import scala.collection.parallel.CollectionConverters._
import Tensor.Storage
import Tensor._
  
def gradientSearch(t: Tensor): Map[Tensor, Storage] = 
  def helper(t: Tensor, accGrad: Storage): List[(Tensor, Storage)] = 
    if t.origin.args.isEmpty && t.hasVar then List((t, accGrad))
    else if !t.hasVar then List()
    else t.origin.args.par.filter(_.hasVar).flatMap(v => helper(v, t.origin.backward(v, accGrad))).toList
  helper(t, Storage.ones(t.storage)).groupMapReduce(_._1)(_._2)(_ + _)
    