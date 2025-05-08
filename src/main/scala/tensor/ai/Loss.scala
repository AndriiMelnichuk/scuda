package scuda.tensor.ai

import scuda.tensor.Tensor
import scuda.tensor.GeneralFunction
import scuda.tensor.storage.Storage
import scuda.tensor.storage.crossEntropyLossGrad

def crossEntropyLoss(prediction: Tensor, target: Tensor): Tensor =

  val res = new Tensor(new GeneralFunction {
    val args: Seq[Tensor] = Seq(prediction)
    val forward: Storage = scuda.tensor.storage.crossEntropyLoss(prediction.storage, target.storage)
    def backward(argument: Tensor, chainGrad: Storage): Storage = 
      crossEntropyLossGrad(prediction.storage, target.storage, chainGrad)
  }, prediction.hasVar)
  res.sum / res.storage.shape(0)

def MSE(prediction: Tensor, target: Tensor): Tensor =
  ((prediction - target).pow(2)).sum / prediction.storage.shape.product