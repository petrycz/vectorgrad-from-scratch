import numpy as np
from vectorgrad.engine import Tensor

def _num_grad(f, A, eps=1e-6):
    g = np.zeros_like(A); it = np.nditer(A, flags=['multi_index'])
    while not it.finished:
        i = it.multi_index; old = A[i]
        A[i] = old + eps; hp = f(); A[i] = old - eps; hm = f(); A[i] = old
        g[i] = (hp - hm) / (2*eps); it.iternext()
    return g

def test_forward_add_mul_matmul():
    A = Tensor([[1.,2.],[3.,4.]]); B = Tensor([[1.,0.],[0.,1.]])
    assert np.allclose((A + B).data, [[2,2],[3,5]])
    assert np.allclose((A * 2).data, [[2,4],[6,8]])
    assert np.allclose((A @ B).data, A.data)

def test_accumulation():
    a = Tensor([3.0]); b = a + a; b.backward()
    assert np.allclose(a.grad, [2.0])

def test_matmul_broadcast_relu_pow_sum_gradcheck():
    rng = np.random.default_rng(0)
    X = Tensor(rng.normal(size=(4,3))); W = Tensor(rng.normal(size=(3,2))); b = Tensor(rng.normal(size=(2,)))
    def scalar(): return ((X @ W + b).relu() ** 2).sum()
    out = scalar()
    for t in (X, W, b): t.grad = np.zeros_like(t.data)
    out.backward()
    gX, gW, gb = X.grad.copy(), W.grad.copy(), b.grad.copy()
    nX = _num_grad(lambda: float(scalar().data), X.data)
    nW = _num_grad(lambda: float(scalar().data), W.data)
    nb = _num_grad(lambda: float(scalar().data), b.data)
    assert np.abs(gX-nX).max() < 1e-5
    assert np.abs(gW-nW).max() < 1e-5
    assert np.abs(gb-nb).max() < 1e-5

def test_tanh_mean_mul_broadcast_gradcheck():
    rng = np.random.default_rng(1)
    A = Tensor(rng.normal(size=(5,4))); c = Tensor(rng.normal(size=(4,)))
    def scalar(): return (A.tanh() * c).mean()
    out = scalar()
    for t in (A, c): t.grad = np.zeros_like(t.data)
    out.backward()
    gA, gc = A.grad.copy(), c.grad.copy()
    nA = _num_grad(lambda: float(scalar().data), A.data)
    nc = _num_grad(lambda: float(scalar().data), c.data)
    assert np.abs(gA-nA).max() < 1e-5 and np.abs(gc-nc).max() < 1e-5
