import numpy as np
from vectorgrad.engine import Tensor
from vectorgrad.nn import MLP

def test_shapes_and_params():
    np.random.seed(0); m=MLP([2,16,16,1])
    assert m(Tensor(np.zeros((5,2)))).data.shape==(5,1)
    assert sum(p.data.size for p in m.parameters())==337

def test_forward_values_match_reference():
    # deterministic weights; expected computed by hand: tanh on hidden, LINEAR on output.
    # Catches: wiring bugs, __call__ logic, wrong activation placement, matmul shape errors.
    m=MLP([2,3,1],act='tanh')
    W1=np.array([[.1,.2,.3],[.4,.5,.6]]); b1=np.array([.01,.02,.03])
    W2=np.array([[.7],[.8],[.9]]);        b2=np.array([.05])
    m.layers[0].W.data=W1.copy(); m.layers[0].b.data=b1.copy()
    m.layers[1].W.data=W2.copy(); m.layers[1].b.data=b2.copy()
    X=np.array([[1.,2.],[-1.,.5]])
    expected=np.tanh(X@W1+b1)@W2+b2          # hidden tanh, output linear
    got=m(Tensor(X)).data
    assert got.shape==(2,1)
    assert np.allclose(got,expected), f"forward mismatch:\n got={got.ravel()}\n exp={expected.ravel()}"

def test_last_layer_is_linear():
    # force a large final pre-activation; a linear output reflects it, a tanh'd one caps near 1.
    m=MLP([1,1,1],act='tanh')
    m.layers[0].W.data=np.array([[0.0]]); m.layers[0].b.data=np.array([1.0])   # hidden pre-act=1
    m.layers[1].W.data=np.array([[10.0]]); m.layers[1].b.data=np.array([5.0])  # out=10*tanh(1)+5 ~ 12.6
    out=m(Tensor(np.array([[0.0]]))).data.item()
    assert abs(out)>1.5, f"output {out:.3f} is squashed -> activation wrongly applied to the LAST layer?"

def test_mlp_gradcheck():
    rng=np.random.default_rng(0); Xv=rng.standard_normal((4,2)); m=MLP([2,3,1],'tanh')
    build=lambda: m(Tensor(Xv)).sum()
    build().backward()
    for p in m.parameters():
        ag=p.grad.copy(); num=np.zeros_like(p.data); fl=p.data.reshape(-1); nf=num.reshape(-1)
        for i in range(fl.size):
            o=fl[i]; fl[i]=o+1e-6; Lp=float(build().data); fl[i]=o-1e-6; Lm=float(build().data); fl[i]=o
            nf[i]=(Lp-Lm)/2e-6
        assert np.allclose(ag,num,atol=1e-4), "MLP param gradient wrong"

def test_mlp_backward_populates_all_param_grads():
    np.random.seed(0); m=MLP([2,8,1]); m.zero_grad()
    m(Tensor(np.random.randn(6,2))).sum().backward()
    for p in m.parameters():
        assert p.grad.shape==p.data.shape and np.any(p.grad!=0)
