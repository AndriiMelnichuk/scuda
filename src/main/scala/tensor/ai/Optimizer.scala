package scuda.tensor.ai

import scuda.tensor.storage.Storage
import scuda.tensor.Tensor
import scala.compiletime.ops.float

type Weight = Storage
type Gradient = Storage
type Optimizer = (Weight, Gradient) => Storage

def SGD(tau: Float): Optimizer =
    (w0, grad) => w0 - (grad * tau)

def momentum(tau: Float, prGrad: Map[Tensor, Storage], mCoef: Float): Optimizer =
    require(mCoef >= 0 && mCoef < 1, s"momentum optimizer: mCoef must be in [0,1), mCoef: $mCoef")
    val prGrads = prGrad.map{ (t, s) => (t.storage, s)}
    (w0, grad) => {
        val ac = prGrads.getOrElse(w0, Storage.zeros(w0))
        w0 - (ac * mCoef + grad * tau)
    }