import numpy as np
from vectorgrad.engine import Tensor
from vectorgrad.nn import MLP
from data import make_moons

def test_shapes_and_params():
    np.random.seed(0)
    model = MLP([2,16,16,1])
    out = model(Tensor(np.zeros((5,2))))
    assert out.data.shape == (5,1)
    n = sum(p.data.size for p in model.parameters())
    assert n == 337            # (2*16+16)+(16*16+16)+(16*1+1)

def test_learns_moons_fast():
    np.random.seed(0)
    X, y01 = make_moons(n=100, noise=0.10, seed=1)
    y = (y01*2-1).reshape(-1,1).astype(float)
    Xt = Tensor(X); model = MLP([2,16,16,1], 'tanh')
    for k in range(100):
        pred = model(Xt)
        loss = (1.0 - Tensor(y)*pred).relu().mean() + 1e-4*sum(((p*p).sum() for p in model.parameters()), Tensor(0.0))
        model.zero_grad(); loss.backward()
        lr = 1.0 - 0.9*k/100
        for p in model.parameters(): p.data -= lr*p.grad
    acc = (np.sign(model(Xt).data) == np.sign(y)).mean()
    assert acc >= 0.95, acc
