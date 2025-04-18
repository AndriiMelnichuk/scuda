package scuda.tensor.storage

import scuda.tensor.cuda.*
import scuda.tensor.cpu.*
import scuda.tensor.cuda.CudaStorage
import scuda.tensor.cpu.ArrayStorage

def relu(x: Storage): Storage = x match
    case x: CudaStorage   => scuda.tensor.cuda.relu(x)
    case x: ArrayStorage  => scuda.tensor.cpu.relu(x)
    case _                => throw new IllegalArgumentException("Unsupported storage type for ReLU")

def reluGrad(pr: Storage, cg: Storage): Storage = (pr, cg) match
    case (x: ArrayStorage, y: ArrayStorage) => scuda.tensor.cpu.reluGrad(x, y)
    case (x: CudaStorage, y: CudaStorage)   => ???
    case _                                  => throw new IllegalArgumentException("Mismatched or unsupported storage types for reluGrad")

def crossEntropyLoss(pr: Storage, target: Storage): Storage = (pr, target) match
    case (x: ArrayStorage, y: ArrayStorage) => scuda.tensor.cpu.crossEntropyLoss(x, y)
    case (x: CudaStorage, y: CudaStorage)   => scuda.tensor.cuda.crossEntropyLoss(x, y)
    case _                                  => throw new IllegalArgumentException("Mismatched or unsupported storage types for crossEntropyLoss")
    
def crossEntropyLossGrad(pr: Storage, target: Storage, chainGrad: Storage): Storage = (pr, target, chainGrad) match
    case (x: ArrayStorage, y: ArrayStorage, z: ArrayStorage) => scuda.tensor.cpu.crossEntropyLossGrad(x, y, z)
    case (x: CudaStorage, y: CudaStorage, z: CudaStorage)    => scuda.tensor.cuda.crossEntropyLossGrad(x, y, z)
    case _                                  => throw new IllegalArgumentException("Mismatched or unsupported storage types for crossEntropyLossGrad")

def stableSoftmax(x: Storage): Storage = x match
    case x: ArrayStorage => scuda.tensor.cpu.stableSoftmax(x) 
    case x: CudaStorage  => scuda.tensor.cuda.stableSoftmax(x)
    case _               => throw new IllegalArgumentException("Mismatched or unsupported storage types for stableSoftmax")

def stableSoftmaxGrad(sm: Storage, cg: Storage): Storage = (sm, cg) match
    case (x: ArrayStorage, y: ArrayStorage) => scuda.tensor.cpu.stableSoftmaxGrad(x, y)
    case (x: CudaStorage, y: CudaStorage)   => scuda.tensor.cuda.stableSoftmaxGrad(x, y)
    case _                                  => throw new IllegalArgumentException("Mismatched or unsupported storage types for crossEntropyLossGrad")

