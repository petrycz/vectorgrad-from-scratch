"""
nn.py — a vectorized neural-net library on your Tensor (GUIDE Stage 4).

A Linear layer is ONE matrix multiply for the whole batch: y = x @ W + b.
Build Linear and MLP; each exposes parameters() and the MLP a zero_grad().
"""
import numpy as np
from .engine import Tensor


class Linear:
    def __init__(self, nin, nout):
        raise NotImplementedError("Stage 4")
    def __call__(self, x):
        raise NotImplementedError("Stage 4")
    def parameters(self):
        raise NotImplementedError("Stage 4")


class MLP:
    def __init__(self, sizes, act='tanh'):
        raise NotImplementedError("Stage 4")
    def __call__(self, x):
        raise NotImplementedError("Stage 4")
    def parameters(self):
        raise NotImplementedError("Stage 4")
    def zero_grad(self):
        raise NotImplementedError("Stage 4")
