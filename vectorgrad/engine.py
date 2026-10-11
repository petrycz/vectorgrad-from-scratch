"""
engine.py — the vectorized autograd core you will build (numpy-backed).

Follow GUIDE.md:
  Stage 1  ->  Tensor.__init__, _unbroadcast, __add__, __mul__   (broadcasting backward)
  Stage 2  ->  __matmul__                                        (dX = dZ @ Wᵀ, dW = Xᵀ @ dZ)
  Stage 3  ->  __pow__, sum, mean, relu, tanh, backward(), conveniences

Tests in tests/test_engine.py go green as you implement each group.
"""
import numpy as np


class Tensor:
    def __init__(self, data, _children=(), _op=''):
        # Stage 1: data = np.asarray(data, float); grad = zeros_like(data);
        #          _backward = no-op; _prev = set(_children); _op = _op
        raise NotImplementedError("Stage 1: see GUIDE.md")

    @staticmethod
    def _unbroadcast(grad, shape):
        # Stage 1: sum `grad` back down to `shape` (the reverse of numpy broadcasting)
        raise NotImplementedError("Stage 1")

    def __add__(self, other):
        raise NotImplementedError("Stage 1")

    def __mul__(self, other):
        raise NotImplementedError("Stage 1")

    def __matmul__(self, other):
        raise NotImplementedError("Stage 2")

    # ---- Stage 3: __pow__, sum, mean, relu, tanh, backward(), and conveniences ----
    def backward(self):
        raise NotImplementedError("Stage 3")
