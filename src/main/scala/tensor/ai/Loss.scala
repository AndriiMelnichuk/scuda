package scuda.tensor.ai

import scuda.tensor.Tensor
import scuda.tensor.GeneralFunction
import scuda.tensor.storage.Storage
import scuda.tensor.storage.crossEntropyLossGrad

type Loss = (Tensor, Tensor) => Tensor

def crossEntropyLoss(prediction: Tensor, target: Tensor): Tensor =

  val res = new Tensor(new GeneralFunction {
    lazy val args: Seq[Tensor] = Seq(prediction)
    lazy val forward: Storage = scuda.tensor.storage.crossEntropyLoss(prediction.storage, target.storage)
    def backward(argument: Tensor, chainGrad: Storage): Storage = 
      crossEntropyLossGrad(prediction.storage, target.storage, chainGrad)
  }, prediction.hasVar)
  res.sum / res.storage.shape(0)
