# Build a Vectorized Autograd Engine — guided capstone

**What you'll build:** the same reverse-mode autodiff as the scalar `micrograd`, but on
**numpy arrays** — a `Tensor` that differentiates array operations, including the two that
matter most in ML: **matrix multiply** and **broadcasting**. Then you'll train an MLP where
each layer is a single `X @ W + b`, and watch it finish in a fraction of a second.

**Prerequisites:** the scalar autograd capstone (`micrograd-from-scratch`), plus two things
from the matrix-calculus module:
- the **chain rule / Jacobian identities** — for `Z = X @ W`, the gradients are
  `dX = dZ @ Wᵀ` and `dW = Xᵀ @ dZ` (you'll derive these below);
- the **layout / transpose** discussion — which transpose goes where is decided by making
  the matrix shapes meet. That rule is the whole content of the matmul backward.

**What's genuinely new vs the scalar engine:** only two ideas — a **broadcasting-aware
backward** (Stage 1) and the **matmul backward** (Stage 2). Everything else (the graph, the
reverse topological sweep, activations, `backward()`) is the micrograd structure you already
built, lifted from scalars to arrays.

**How to work:** same as before — concept, your task, code, checkpoint. The suite starts red.
`SOLUTION/` has a reference; write your own first.

---

## Stage 0 — Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                                            # fails — your to-do list
```

---

## Stage 1 — The `Tensor` node, and broadcasting backward

**Concept.** A `Tensor` wraps a numpy array and, like `Value`, records how it was made.
Elementwise `+` and `*` look just like micrograd — but numpy **broadcasts**: adding a bias
of shape `(nout,)` to a batch of shape `(N, nout)` copies the bias across all `N` rows.
Forward copies; therefore **backward must sum**. A parameter that was broadcast to many
positions receives a gradient contribution from each, and — exactly the "sum the routes"
rule of the multivariable chain rule — those contributions add. The helper `_unbroadcast`
sums a gradient back down to the original shape.

**Your task.** Implement `__init__`, `_unbroadcast`, `__add__`, `__mul__`.

**Code.**

```python
import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @staticmethod
    def _unbroadcast(grad, shape):
        # reverse numpy broadcasting: collapse grad back to `shape`
        while grad.ndim > len(shape):           # extra leading axes -> sum them off
            grad = grad.sum(axis=0)
        for i, s in enumerate(shape):           # axes that were size-1 -> sum, keepdim
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
```

**Checkpoint.** `pytest -q tests/test_engine.py::test_forward_add_mul_matmul` (the add/mul
parts) and `::test_accumulation`.

---

## Stage 2 — `matmul`, the heart of the engine

**Concept & derivation.** For `Z = X @ W` with `X: (N, d_in)`, `W: (d_in, d_out)`,
`Z: (N, d_out)`, each entry is `Z[n,j] = Σ_k X[n,k] W[k,j]`. Given the upstream gradient
`dZ = ∂L/∂Z`, the chain rule gives

- `∂L/∂X[n,k] = Σ_j dZ[n,j] · W[k,j]`  →  `dX = dZ @ Wᵀ`
- `∂L/∂W[k,j] = Σ_n X[n,k] · dZ[n,j]`  →  `dW = Xᵀ @ dZ`

These are the matrix-calculus identities, and the transposes are exactly what the shapes
force: `dX` must be `(N, d_in)`, so `dZ (N,d_out) @ Wᵀ (d_out,d_in)` is the only product that
fits; likewise `dW` must be `(d_in, d_out) = Xᵀ (d_in,N) @ dZ (N,d_out)`. The layout rule
from the module — *insert the transpose that makes the dimensions meet* — picks both.

**Your task.** Implement `__matmul__`.

**Code.**

```python
    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other), '@')
        def _backward():
            self.grad  = self.grad  + out.grad @ other.data.T   # dX = dZ @ Wᵀ
            other.grad = other.grad + self.data.T @ out.grad     # dW = Xᵀ @ dZ
        out._backward = _backward
        return out
```

**Checkpoint.** After Stage 3 you'll run the full gradient check; for now confirm the
forward matmul in `::test_forward_add_mul_matmul` passes.

---

## Stage 3 — Reductions, activations, and `backward()`

**Concept.** A loss is a scalar, so you need a reduction (`sum`/`mean`) to collapse the
batch. `sum`'s backward broadcasts the upstream scalar back over every summed element
(each contributed once, local derivative 1). Activations are elementwise, so their backward
is a diagonal multiply — the same `relu`/`tanh` as micrograd. `backward()` is identical to
the scalar version: reverse-topological order, seed the output grad to ones, run closures.

**Your task.** Implement `__pow__`, `sum`, `mean`, `relu`, `tanh`, `backward`, and the
conveniences (`__neg__`, `__sub__`, `__rsub__`, `__radd__`, `__rmul__`, `__truediv__`,
`shape`, `__repr__`). See `SOLUTION/engine.py` for the exact convenience signatures.

**Code (the core pieces).**

```python
    def __pow__(self, k):
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
```

**Checkpoint — the real test of correctness:**

```bash
pytest -q tests/test_engine.py     # finite-difference gradient checks for matmul + broadcasting
```

If the gradient checks pass to ~1e-5, your chain rule through matrix multiply and
broadcasting is correct.

---

## Stage 4 — Layers as matrix multiplies

**Concept.** This is the contrast that makes the whole capstone. In the scalar engine a
layer was `nout` neurons, each a Python loop of `Value` multiplies — thousands of objects.
Here a layer is **one array and one matmul**: weights `W: (nin, nout)`, bias `b: (nout,)`,
and `forward = x @ W + b` for the *entire batch at once*. The bias broadcasts over the
batch (Stage 1 handles its gradient).

**Your task.** Implement `Linear` and `MLP` in `vectorgrad/nn.py`.

**Code.**

```python
import numpy as np
from .engine import Tensor

class Linear:
    def __init__(self, nin, nout):
        self.W = Tensor(np.random.randn(nin, nout) / np.sqrt(nin))
        self.b = Tensor(np.zeros(nout))
    def __call__(self, x):
        return x @ self.W + self.b
    def parameters(self):
        return [self.W, self.b]

class MLP:
    def __init__(self, sizes, act='tanh'):
        self.layers = [Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]
        self.act = act
    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = x.tanh() if self.act == 'tanh' else x.relu()
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)
```

**Checkpoint.** `pytest -q tests/test_nn.py::test_shapes_and_params`

---

## Stage 5 — Train it (whole dataset as one matrix)

**Concept.** The data is now one array `X: (N, 2)`, and the entire forward pass over all N
points is a few matmuls. The loss, gradient, and update are the optimization loop from the
course — but every step touches the whole batch at once.

**Your task.** Write `train.py` (core below; full version with plotting in `SOLUTION/`).

```python
import numpy as np
from vectorgrad.engine import Tensor
from vectorgrad.nn import MLP
from data import make_moons

def forward_loss(model, Xt, y, alpha=1e-4):
    pred = model(Xt)                                     # (N,1) — one pass
    data_loss = (1.0 - Tensor(y) * pred).relu().mean()   # hinge
    reg = alpha * sum(((p*p).sum() for p in model.parameters()), Tensor(0.0))
    acc = (np.sign(pred.data) == np.sign(y)).mean()
    return data_loss + reg, acc

def main():
    np.random.seed(0)
    X, y01 = make_moons(n=100, noise=0.10, seed=1)
    y = (y01*2 - 1).reshape(-1, 1).astype(float)
    Xt, model = Tensor(X), MLP([2, 16, 16, 1], 'tanh')
    for k in range(100):
        loss, acc = forward_loss(model, Xt, y)
        model.zero_grad(); loss.backward()
        lr = 1.0 - 0.9*k/100
        for p in model.parameters():
            p.data -= lr * p.grad
        if k % 10 == 0:
            print(f"step {k:3d}  loss {loss.data.item():.4f}  acc {acc*100:.1f}%")
```

**Checkpoint.** `pytest -q tests/test_nn.py::test_learns_moons_fast` and `python train.py`
(→ 100%, boundary.png).

---

## Stage 6 — Feel the speedup (the actual lesson)

Time 100 steps here vs the scalar capstone on the *same* problem:

| engine | 100 steps | why |
|---|---|---|
| scalar `micrograd` | ~160 s | each number is a Python object; ~10⁴ objects/step |
| vectorized `vectorgrad` | ~0.04 s | each layer is one `@`; numpy/BLAS runs it near hardware speed |

Roughly a **1000–4000× speedup** with identical math. This is the thesis in one experiment:
linear algebra is useful in modern tech because the operations compile down to dense
matrix multiplies that hardware executes at full throughput — the reason GPUs exist, the
reason deep learning is feasible. (Add a line to `train.py` that prints wall-clock per step
and confirm it yourself.)

---

## Stage 7 — Ship it

```bash
git init && git add . && git commit -m "Vectorized autograd engine: Tensor + matmul/broadcast backward, MLP on two-moons"
git remote add origin git@github.com:<you>/vectorgrad-from-scratch.git
git branch -M main && git push -u origin main
```

## Stretch goals (your own work)

1. **Minibatching.** Sample rows of `X` each step; confirm it still converges. (Sets up SGD
   for larger datasets.)
2. **A harder dataset.** Add `make_spiral` to `data.py` and classify it with a deeper MLP.
3. **Adam.** Implement the Adam optimizer (per-parameter running averages) and compare its
   convergence to plain SGD.
4. **Softmax + cross-entropy.** Add `exp`, `log`, and a numerically-stable softmax; switch to
   a 2-logit output with cross-entropy loss. (This is the exact loss the GPT capstone uses —
   and the first place probability shows up. You'll build the theory in the probability
   course; here you can implement it mechanically.)
5. **Check against PyTorch.** Rebuild one `X @ W + b -> relu -> loss` with `torch`, call
   `.backward()`, and confirm your `dW`, `dX` match.

## How the code maps to the math

| op (code) | backward | from the module |
|---|---|---|
| `X @ W` | `dX = dZ @ Wᵀ`, `dW = Xᵀ @ dZ` | chain rule + the transpose chosen by dimension-matching |
| broadcast `+ b` | sum grad over the broadcast axes | multivariable chain rule: add the routes |
| `relu` / `tanh` | diagonal multiply by the slope | elementwise ⇒ diagonal Jacobian |
| `sum` / `mean` | broadcast the scalar grad back | each element contributes once (local deriv 1) |
| `backward()` | reverse-topo product of locals | backprop = Jacobian product, right-to-left from the scalar loss |
