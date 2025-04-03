package scuda.tensor.ai

import scuda.tensor.Storage

type Weight = Storage
type Gradient = Storage
type Optimizer = (Weight, Gradient) => Storage

def SGD(tau: Float): Optimizer =
    (w0, grad) => (grad * -1) * tau + w0
