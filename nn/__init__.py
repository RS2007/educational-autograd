import numpy as np
from tensor.tensor import Value


class Linear:
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

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        return np.hstack([*self._w, self._b])
