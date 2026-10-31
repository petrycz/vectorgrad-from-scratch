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
        self.data = np.asarray(data, dtype=np.float64)
        self._children = _children
        self.grad = np.zeros_like(self.data)
        # Stage 1: data = np.asarray(data, float); grad = zeros_like(data);
        #          _backward = no-op; _prev = set(_children); _op = _op
        self._op = _op
        self._backward = lambda:None

    @staticmethod
    def _unbroadcast(grad, shape):
        # Stage 1: sum `grad` back down to `shape` (the reverse of numpy broadcasting)
        while grad.ndim > len(shape):
            #numpy always prepends new axis so we are just summing them back down using axis 0 repeatedly
            grad = grad.sum(axis=0)

        for i,j in enumerate(shape):
            if j==1:
                grad.sum(axis=i)

        return grad

#need ot add the backward bit onto the result
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        dat = self.data + other.data
        res = Tensor(dat, {self, other})
        def _back():
            self.grad+=self._unbroadcast(res.grad, self.data.shape)
            other.grad+=self._unbroadcast(res.grad, other.data.shape)
        res._backward = _back
        return res

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        dat = self.data*other.data
        res = Tensor(dat, {self, other})
        def _back():
            self.grad+=self._unbroadcast(res.grad*other.data, self.data.shape)
            other.grad+=self._unbroadcast(res.grad*self.data, other.data.shape)
        res._backward = _back
        return res

    def __matmul__(self, other):
        # we use row picture for the derivatives. 
        other = other if isinstance(other, Tensor) else Tensor(other)
        dat = self.data@other.data if self.data.ndim!=0 and other.data.ndim!=0 else self.data*other.data
        res = Tensor(dat, {self, other})
        def _back():
            # we are not unbroadcasting because there is no sense in which we would "broadcast" matrix multiplication
            self.grad+=res.grad@np.transpose(other.data)
            other.grad += np.transpose(self.data)@res.grad
        res._backward=_back
        return res

    # ---- Stage 3: __pow__, sum, mean, relu, tanh, backward(), and conveniences ----
    def __pow__(self, n):
        res = Tensor(self.data**n, {self})
        def _back():
            self.grad += res.grad*n*self.data**(n-1)
        res._backward = _back
        return res

    def sum(self, axis=None, keepdims=False):
        res = Tensor(self.data.sum(axis=axis, keepdims=keepdims), {self})
        def _back():
            g = res.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad+= res.grad*np.ones_like(self.data)
        res._backward = _back
        return res

    def mean(self):
        res = self.sum()*(1.0/self.data.size)
        return res

    def srelu(self):
        res = Tensor(np.maximum(0, self.data), {self})
        def _back():
            self.grad+=res.grad*(self.data>0)
        res._backward = _back
        return res

    def relu(self):
        out = Tensor(np.maximum(0.0, self.data), (self,), 'relu')
        def _backward():
            self.grad = self.grad + (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        res = Tensor(t, {self})
        def _back():
            self.grad+=res.grad*(1-t**2)
        res._backward = _back
        return res
        
    def backward(self):
        # seed the grad
        #create final loss tensor
        #dfs _backward() calls
        nodes = []
        final_l = self.sum()
        final_l.grad = 1.0
        seen = set()
        curr = final_l
        def trav(node):
            seen.add(node)
            for p in node._children:
                if not p in seen:
                    trav(p)
            nodes.append(node)
        trav(curr)
        for n in reversed(nodes):
            n._backward()

    def __neg__(self):      return self * -1          # unary minus: -a  == a * -1
    def __sub__(self, o):   return self + (-o)         # a - b  ==  a + (-b)
    def __rsub__(self, o):  return (-self) + o 



