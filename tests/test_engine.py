"""
Engine tests, grouped by GUIDE stage. Each group actually EXERCISES what that
stage builds — including the backward closures (via out._backward()) before
backward() itself exists, so a missing _backward or missing _unbroadcast fails here.
"""
import numpy as np
from vectorgrad.engine import Tensor

def _close(a, b, tol=1e-6):
    return np.allclose(np.asarray(a, float), np.asarray(b, float), atol=tol, rtol=1e-5)


# ======================= STAGE 1: +, *, broadcasting =======================

def test_forward_add_mul():
    # Stage 1 forward VALUES: elementwise + and *, with broadcasting
    a = Tensor([[1., 2.], [3., 4.]])
    b = Tensor([[10., 20.], [30., 40.]])
    assert _close((a + b).data, [[11, 22], [33, 44]])
    assert _close((a * b).data, [[10, 40], [90, 160]])
    assert _close((a + Tensor([100., 200.])).data, [[101, 202], [103, 204]])   # broadcast

def test_forward_matmul():
    # Stage 2 forward: matmul
    a = Tensor([[1., 2.], [3., 4.]])
    W = Tensor([[1., 0., 1.], [0., 1., 1.]])
    assert _close((a @ W).data, np.array([[1, 2], [3, 4]]) @ np.array([[1, 0, 1], [0, 1, 1]]))

def test_add_backward_and_unbroadcast():
    # drives the + closure directly; catches a missing _backward AND a missing _unbroadcast
    A = Tensor([[1., 2., 3.], [4., 5., 6.]])   # (2,3)
    b = Tensor([10., 20., 30.])                # (3,) -> broadcast across rows
    out = A + b
    out.grad = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    out._backward()
    assert A.grad.shape == (2, 3), "no gradient on A -> did you attach out._backward in __add__?"
    assert _close(A.grad, out.grad), "A matches Z's shape: gradient must pass straight through"
    assert b.grad.shape == (3,), "bias grad must be (3,) -> _unbroadcast missing in __add__?"
    assert _close(b.grad, [0.5, 0.7, 0.9]), "broadcast bias grad must SUM down the copied axis"

def test_mul_backward_and_unbroadcast():
    A = Tensor([[1., 2., 3.], [4., 5., 6.]])
    b = Tensor([10., 20., 30.])
    out = A * b
    out.grad = np.ones((2, 3))
    out._backward()
    assert _close(A.grad, [[10, 20, 30], [10, 20, 30]]), "dA = b (broadcast) * dOut"
    assert b.grad.shape == (3,), "_unbroadcast missing in __mul__?"
    assert _close(b.grad, [1 + 4, 2 + 5, 3 + 6]), "db sums A*dOut down the copied axis"

def test_accumulation():
    # a node used twice must ACCUMULATE (+=), not overwrite (=)
    a = Tensor([3.0])
    out = a * a                      # both operands are the same node
    out.grad = np.array([1.0])
    out._backward()
    assert _close(a.grad, 6.0), f"d(a*a)/da = 2a = 6; got {a.grad} (did you use '=' instead of '+=' ?)"


# ======================= STAGE 2: matmul backward =======================

def test_matmul_backward():
    X = Tensor([[1., 2.], [3., 4.], [5., 6.]])      # (3,2)
    W = Tensor([[1., 0., 1.], [0., 1., 1.]])        # (2,3)
    Z = X @ W                                        # (3,3)
    assert Z.data.shape == (3, 3)
    dZ = (np.arange(9).reshape(3, 3) + 1) * 0.1
    Z.grad = dZ.copy()
    Z._backward()
    assert X.grad.shape == (3, 2), "no dX -> attach out._backward in __matmul__?"
    assert _close(X.grad, dZ @ W.data.T), "dX must be dZ @ Wᵀ"
    assert _close(W.grad, X.data.T @ dZ), "dW must be Xᵀ @ dZ"


# ======================= STAGE 3: backward(), reductions, activations, gradcheck =======================

def test_backward_full_graph():
    # backward() must seed the end and propagate through the whole graph
    a, b = Tensor([2.0]), Tensor([3.0])
    (a * b + a).backward()                 # L = a*b + a
    assert _close(a.grad, 4.0), "dL/da = b + 1 = 4"
    assert _close(b.grad, 2.0), "dL/db = a = 2"

def test_sum_and_mean_backward():
    x = Tensor([[1., 2.], [3., 4.]]); x.sum().backward()
    assert _close(x.grad, np.ones((2, 2))), "each summed element contributes 1"
    y = Tensor([[1., 2.], [3., 4.]]); y.mean().backward()
    assert _close(y.grad, np.ones((2, 2)) / 4.0), "mean backward = sum/ N"

def test_relu_tanh_backward():
    x = Tensor([-2.0, 0.5, 3.0]); x.relu().sum().backward()
    assert _close(x.grad, [0., 1., 1.]), "relu grad is the 0/1 gate"
    y = Tensor([-1.0, 0.0, 2.0]); y.tanh().sum().backward()
    assert _close(y.grad, 1 - np.tanh([-1.0, 0.0, 2.0]) ** 2), "tanh grad is 1 - tanh^2"

def _numgrad(build, arr, eps=1e-6):
    g = np.zeros_like(arr); flat = arr.reshape(-1); gf = g.reshape(-1)
    for i in range(flat.size):
        old = flat[i]
        flat[i] = old + eps; Lp = float(build().data)
        flat[i] = old - eps; Lm = float(build().data)
        flat[i] = old
        gf[i] = (Lp - Lm) / (2 * eps)
    return g

def test_gradcheck_through_matmul_broadcast_activation():
    # the real correctness proof: finite differences vs analytic, through @, +bias, relu, **2, sum
    rng = np.random.default_rng(0)
    Xv = rng.standard_normal((4, 3)); Wv = rng.standard_normal((3, 2)); bv = rng.standard_normal((2,))
    tensors = {}
    def build():
        X, W, b = Tensor(Xv), Tensor(Wv), Tensor(bv)
        tensors['X'], tensors['W'], tensors['b'] = X, W, b
        return (((X @ W) + b).relu() ** 2).sum()
    build().backward()
    X, W, b = tensors['X'], tensors['W'], tensors['b']
    assert _close(W.grad, _numgrad(build, Wv), tol=1e-4), "dW wrong (check matmul backward)"
    assert _close(b.grad, _numgrad(build, bv), tol=1e-4), "db wrong (check broadcast/unbroadcast)"
    assert _close(X.grad, _numgrad(build, Xv), tol=1e-4), "dX wrong (check matmul backward)"

def test_gradcheck_tanh_mean():
    rng = np.random.default_rng(1)
    Xv = rng.standard_normal((3, 2)); Wv = rng.standard_normal((2, 2))
    tensors = {}
    def build():
        X, W = Tensor(Xv), Tensor(Wv)
        tensors['W'] = W
        return (X @ W).tanh().mean()
    build().backward()
    assert _close(tensors['W'].grad, _numgrad(build, Wv), tol=1e-4)
