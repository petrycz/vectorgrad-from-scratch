"""
nn.py — a vectorized neural-net library on your Tensor (GUIDE Stage 4).

A Linear layer is ONE matrix multiply for the whole batch: y = x @ W + b.
Build Linear and MLP; each exposes parameters() and the MLP a zero_grad().
"""
import numpy as np
from .engine import Tensor


class Linear:
    def __init__(self, nin, nout):
        self.W = Tensor(np.random.randn(nin, nout) / np.sqrt(nin))
        self.b = Tensor(np.zeros(nout))

    def __call__(self, x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        print("x and w:", x.data.shape, self.W.data.shape)
        z = x@self.W + self.b
        return z

    def parameters(self):
        return [self.W, self.b]


class MLP:
    def __init__(self, sizes, act='tanh'):
        #important - keep in mind the first sizes entry has to match the input size of X and the last one has to match the expected output (usually 1 node for a binary classification problem)
        self.sizes = sizes
        self.act = act
        self.layers = [Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]

    def __call__(self, x):
        # iterate through each layer, with the input to each new one being the output from the previous passed through the activation function
        #forward pass
        inp = x
        for i, l in enumerate(self.layers):
            inp = l(inp)
            if i<len(self.layers) -1:
                inp = inp.tanh() if self.act=='tanh' else inp.relu()

        return inp

    def parameters(self):
        params = []
        for l in self.layers:
            params.extend(l.parameters())

        return params

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

