"""Train the vectorized MLP on two-moons, time it, and save the decision boundary."""
import time, numpy as np
from vectorgrad.engine import Tensor
from vectorgrad.nn import MLP
from data import make_moons

def forward_loss(model, Xt, y, alpha=1e-4):
    pred = model(Xt)                                   # (N,1) — one pass, whole batch
    data_loss = (1.0 - Tensor(y) * pred).relu().mean() # hinge, vectorized
    reg = alpha * sum(((p * p).sum() for p in model.parameters()), Tensor(0.0))
    acc = (np.sign(pred.data) == np.sign(y)).mean()
    return data_loss + reg, acc

def main():
    np.random.seed(0)
    X, y01 = make_moons(n=100, noise=0.10, seed=1)
    y = (y01 * 2 - 1).reshape(-1, 1).astype(float)
    Xt = Tensor(X)
    model = MLP([2, 16, 16, 1], act='tanh')
    STEPS = 100
    t0 = time.time()
    for k in range(STEPS):
        loss, acc = forward_loss(model, Xt, y)
        model.zero_grad(); loss.backward()
        lr = 1.0 - 0.9 * k / STEPS
        for p in model.parameters():
            p.data -= lr * p.grad
        if k % 10 == 0 or k == STEPS - 1:
            print(f"step {k:3d}   loss {loss.data.item():.4f}   acc {acc*100:.1f}%")
    print(f"\n{STEPS} steps in {time.time()-t0:.3f}s   final acc {acc*100:.1f}%")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        h = 0.25
        xx, yy = np.meshgrid(np.arange(X[:,0].min()-1, X[:,0].max()+1, h),
                             np.arange(X[:,1].min()-1, X[:,1].max()+1, h))
        pts = np.c_[xx.ravel(), yy.ravel()]
        Z = (model(Tensor(pts)).data > 0).reshape(xx.shape)      # grid in one matmul
        plt.figure(figsize=(5,5))
        plt.contourf(xx, yy, Z, cmap="RdBu", alpha=0.3)
        plt.scatter(X[:,0], X[:,1], c=y01, cmap="RdBu", edgecolors="k", s=28)
        plt.title("vectorized MLP — decision boundary")
        plt.tight_layout(); plt.savefig("boundary.png", dpi=110)
        print("saved boundary.png")
    except Exception as e:
        print("plot skipped:", e)

if __name__ == "__main__":
    main()
