import numpy as np
from .engine import Tensor

class Linear:
    """A fully-connected layer: y = x @ W + b.  One matrix multiply for the whole batch."""
    def __init__(self, nin, nout):
        self.W = Tensor(np.random.randn(nin, nout) / np.sqrt(nin))  # scaled init
        self.b = Tensor(np.zeros(nout))
    def __call__(self, x):            # x: (N, nin) Tensor  ->  (N, nout) Tensor
        return x @ self.W + self.b
    def parameters(self):
        return [self.W, self.b]

class MLP:
    def __init__(self, sizes, act='tanh'):
        self.layers = [Linear(sizes[i], sizes[i+1]) for i in range(len(sizes) - 1)]
        self.act = act
    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:                 # no activation on the output
                x = x.tanh() if self.act == 'tanh' else x.relu()
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)
