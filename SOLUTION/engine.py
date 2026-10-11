import numpy as np

class Tensor:
    """An n-d array and its gradient — one node in the computation graph."""
    def __init__(self, data, _children=(), _op=''):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @staticmethod
    def _unbroadcast(grad, shape):
        # reverse numpy broadcasting: sum grad back down to `shape`
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        for i, s in enumerate(shape):
            if s == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad  = self.grad  + Tensor._unbroadcast(out.grad, self.data.shape)
            other.grad = other.grad + Tensor._unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad  = self.grad  + Tensor._unbroadcast(other.data * out.grad, self.data.shape)
            other.grad = other.grad + Tensor._unbroadcast(self.data  * out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other), '@')
        def _backward():
            self.grad  = self.grad  + out.grad @ other.data.T   # dX = dZ @ Wᵀ
            other.grad = other.grad + self.data.T @ out.grad     # dW = Xᵀ @ dZ
        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float))
        out = Tensor(self.data ** k, (self,), f'**{k}')
        def _backward():
            self.grad = self.grad + (k * self.data ** (k - 1)) * out.grad
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), 'sum')
        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad = self.grad + np.ones_like(self.data) * g
        out._backward = _backward
        return out

    def mean(self):
        return self.sum() * (1.0 / self.data.size)

    def relu(self):
        out = Tensor(np.maximum(0.0, self.data), (self,), 'relu')
        def _backward():
            self.grad = self.grad + (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), 'tanh')
        def _backward():
            self.grad = self.grad + (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for c in v._prev:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def __neg__(self):        return self * -1.0
    def __radd__(self, o):    return self + o
    def __sub__(self, o):     return self + (-o if isinstance(o, Tensor) else Tensor(o)*-1)
    def __rsub__(self, o):    return (self * -1) + o
    def __rmul__(self, o):    return self * o
    def __truediv__(self, o): return self * (o ** -1 if isinstance(o, Tensor) else Tensor(o) ** -1)
    @property
    def shape(self): return self.data.shape
    def __repr__(self): return f"Tensor(shape={self.data.shape}, op={self._op!r})"
