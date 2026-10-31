"""train.py — train the vectorized MLP on two-moons, time it, draw the boundary (Stages 5-6).

Reference in SOLUTION/train.py if you get stuck.
"""
from data import make_moons
from vectorgrad.nn import MLP
from vectorgrad.engine import Tensor

def get_loss(scores, y):
    inp = Tensor(1.0)+ (-scores*y)
    outp = inp.relu()
    return outp


def main():
    # create the mlp
    sizes = [2,13,13,1]
    step_size = 0.50
    reps = 100
    mlp = MLP(sizes)
    step_size = 0.50
    X, y = make_moons(100)
    y = (y*2 - 1).reshape(-1, 1).astype(float)
    for rep in range(reps):
        step_size = max(0.1, step_size-0.01) if rep%10==0 else step_size
        scores = mlp(X)
        loss_vec = get_loss(scores, y)
        loss = loss_vec.sum()
        mlp.zero_grad()
        loss.backward()
        for p in mlp.parameters():
            p.data -= p.grad*step_size*(1.0/len(X))

        print("loss:", loss.data)

    import matplotlib
    matplotlib.use("Agg")                  # writes a file; delete this line for a popup window
    import matplotlib.pyplot as plt
    import numpy as np

    # build a grid of points covering the data
    h = 0.25
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    pts = np.c_[xx.ravel(), yy.ravel()]                   # all grid points, shape (G, 2)

    # classify the ENTIRE grid in ONE forward pass (the vectorgrad payoff)
    Z = (mlp(Tensor(pts)).data > 0).reshape(xx.shape)   # True/False -> class, back to grid shape

    plt.figure(figsize=(5, 5))
    plt.contourf(xx, yy, Z, cmap="RdBu", alpha=0.3)                        # shaded decision regions
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolors="k", s=28)  # the actual data points
    plt.title("vectorized MLP — decision boundary")
    plt.tight_layout()
    plt.savefig("boundary.png", dpi=110)
    print("saved boundary.png")

if __name__ == "__main__":
    main()
