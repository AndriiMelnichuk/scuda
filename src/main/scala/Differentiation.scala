package scuda

import scala.collection.parallel.CollectionConverters._
import Tensor.Storage
import Tensor._

// /* 
//   @retutn paths to all var that needs grad
// */
// def pathsSearcher(t: Tensor): List[List[GeneralFunction]] = 
//   def dfs(node: Tensor, path: List[GeneralFunction]): List[List[GeneralFunction]] = 
//     node.origin match 
//       case v if v.args.length == 0 && node.isGrad => List(path :+ v)
//       case v if v.args.length == 0 => List()
//       case v => v.args.par.map(x => dfs(x, path :+ v)).reduce( _ ::: _).seq
//   dfs(t, List())
  
// /* 
//   map funtions to dirivation function
//  */
// def evaluator(a: List[GeneralFunction]): List[(() => Storage, Storage => Storage)] =
//   def helper(a: List[GeneralFunction], acc:  List[(() => Storage, Storage => Storage)]):  List[(() => Storage, Storage => Storage)] =
//     a.head match
//       case v if v.args.length == 0 => acc 
//       // case x if x.args.length == 1 => helper(a.tail, acc :+ (x.backward(0), x.reducer(0))) 
//       case x => 
//         val i = x.args.indexWhere(_.origin == a.tail.head)
//         helper(a.tail, acc :+ (x.backward(i), x.reducer(0)))
//   helper(a, List())

// /* 
//   main function for gradient search
//  */
// def gradientSearch(t: Tensor) =
//   val paths = pathsSearcher(t)
//   paths.par
//   .map(path => (
//     path.last, 
//     evaluator(path).par.map((gr, reducer) => (gr(), reducer)).seq.reduceLeft((acc, x) => (x._2(acc._1), acc._2))
//   ))
//   .map(x => (x._1, x._2._1))
//   .groupBy(_._1)
//   .map {case (k, v) => (k, v.map(_._2).reduce(_ + _))}
//   .seq
  
def gradientSearch(t: Tensor) = 
  def helper(t: Tensor, accGrad: Storage): List[(Tensor, Storage)] = 
    if t.origin.args.isEmpty && t.hasVar then List((t, accGrad))
    else if !t.hasVar then List()
    else t.origin.args.par.filter(_.hasVar).flatMap(v => helper(v, t.origin.backward(v, accGrad))).toList
  helper(t, Storage.ones(t.storage))
    