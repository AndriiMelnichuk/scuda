package scuda.tensor.ai

import scuda.tensor.storage.Storage

type Weight = Storage
type Gradient = Storage
type Optimizer = (Weight, Gradient) => Storage

def SGD(tau: Float): Optimizer =
    (w0, grad) => -grad * tau + w0
