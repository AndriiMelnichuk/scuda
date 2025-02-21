package scuda

import scala.collection.parallel.CollectionConverters._


/* 
  @retutn paths to all var that needs grad
*/
def pathsSearcher(t: Tensor): List[List[GeneralFunction]] = 

  def dfs(node: Tensor, path: List[GeneralFunction]): List[List[GeneralFunction]] = 
    node.origin match 
      case create: CreateTensor if create.isGrad => List(path :+ create)
      case _: CreateTensor => List()
      case v => v.args.par.map(x => dfs(x, path :+ v)).reduce( _ ::: _).seq
      // case func1: Funtion1arg  => dfs(func1.a, path :+ func1)
      // case func2: Funtion2arg  => dfs(func2.a, path :+ func2) ::: dfs(func2.b, path :+ func2) // TODO paralel desigion

  dfs(t, List())
  
/* 
  map funtions to dirivation function
 */
def evaluator(a: List[GeneralFunction]) =
  def helper(a: List[GeneralFunction], acc: List[() => Storage]): List[() => Storage] =
    a.head match
      case _: CreateTensor => acc 
      case x if x.args.length == 1 => helper(a.tail, acc :+ x.backward(0)) 
      case x => 
        val i = x.args.indexWhere(_.origin == a.tail.head)
        helper(a.tail, acc :+ x.backward(i))
      
      // case v: Funtion1arg => helper(a.tail, acc :+ {() => v.grad})
      // case v: Funtion2arg => helper(a.tail, acc :+ {() => if v.a.origin == a.tail.head then v.gradLeft else v.gradRight})
  helper(a, List())

/* 
  main function for gradient search
 */
def gradientSearch(t: Tensor) =
  val paths = pathsSearcher(t)
  paths.par.map(v => (v.last, evaluator(v).par.map(_()).reduce(_ * _)))
  .groupBy(_._1)
  .map {case (k, v) => (k, v.map(_._2).reduce(_ + _))}
  .seq

