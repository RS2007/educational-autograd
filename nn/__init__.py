import numpy as np
from tensor.tensor import Tensor, Value

# TODO: Refactor to use the tensor abstraction


class Module:
    def __init__(self):
        pass

    def __call__(self, x):
        return self.forward(x)


class Linear(Module):
    def __init__(self, in_feat, out_feat, bias=True):
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bias = bias
        self._w = np.array(
            [[Value(elem) for elem in row] for row in np.random.rand(out_feat, in_feat)]
        )
        self._b = np.array([Value(x) for x in np.random.rand(out_feat)])

    def __repr__(self):
        return (
            f"in_feat: {self.in_feat} and out_feat:{self.out_feat} with " + "bias"
            if self.bias
            else "no bias"
        )

    def weights(self):
        return np.array([[elem._val for elem in row] for row in self._w])

    def biases(self):
        return np.array([x._val for x in self._b])

    def forward(self, x):
        return (self._w) @ x + self._b


class Linear2(Module):
    def __init__(self, in_feat, out_feat, bias=True):
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bias = bias
        self.W = Tensor(np.random.rand(out_feat, in_feat))
        self.b = Tensor(np.random.rand(out_feat))

    def forward(self, x):
        return x @ self.W.transpose() + self.b

    def parameters(self):
        return [self.W, self.b]
