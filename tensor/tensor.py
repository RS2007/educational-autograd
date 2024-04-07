import numpy as np


class Tensor:
    # TODO: Unfinished abstraction, need to make lots of changes
    def __init__(self, arr, parents=None, op=None, requires_grad=False):
        self.data = np.array(arr)
        self.requires_grad = requires_grad
        self.grad = None
        self.op = op if op is not None else None
        self.parents = parents if parents is not None else None
        self._backward = lambda: None

    def __add__(self, b):
        val = Plus.forward(self, b)
        val.op = Plus.backward
        val.parents = [self, b]
        return val

    def __mul__(self, b):
        raise NotImplementedError

    def __repr__(self):
        return f"data: {self.data} and grad: {self.grad} with op: {self.op}"

    def __pow__(self, n):
        raise NotImplementedError

    def __truediv__(self, b):
        raise NotImplementedError

    def sum(self):
        val = Sum.forward(self)
        val.op = Sum.backward
        val.parents = [self]
        return val

    def mean(self):
        raise NotImplementedError

    def backward(self):
        if self.op is None:
            return

        if self.grad is None:
            assert (
                self.data.shape == ()
            ), f"Backward should be called only for scalar tensors, got shape {self.data.shape}"
            self.grad = Tensor([1.0])
        visited = {}
        queue = []

        def _topo(graph):
            if graph in visited:
                return
            if graph.parents is not None:
                for parent in graph.parents:
                    _topo(parent)
            queue.append(graph)
            visited[graph] = True

        _topo(self)
        for node in reversed(queue):
            if node.op is None:
                continue
            grads = node.op(node.parents, node.grad)
            if len(node.parents) == 1:
                grads = [grads]
            for tensor, grad in zip(node.parents, grads):
                if grad is None:
                    continue
                if tensor.grad is None:
                    tensor.grad = Tensor(np.zeros_like(tensor.data).astype(np.float32))
                tensor.grad.data += grad.data


class Plus:
    @staticmethod
    def forward(*args):
        assert len(args) == 2
        [a, b] = args
        return Tensor(a.data + b.data)

    @staticmethod
    def backward(parents, grad):
        x, y = parents
        if isinstance(grad, Tensor):
            return grad, grad
        return Tensor([grad]), Tensor([grad])


class Sum:
    @staticmethod
    def forward(*args):
        assert len(args) == 1
        return Tensor(np.sum(args[0].data))

    @staticmethod
    def backward(parents, grad):
        assert len(parents) == 1
        (parent,) = parents
        return Tensor(np.broadcast_to(grad.data, parent.data.shape))


class Value:
    def __init__(self, val, parents=None, op=None):
        self._val = val
        self._grad = 0.0
        self._parents = None if parents is None else np.array([*parents])
        self._op = None if op is None else op
        self._backward = lambda: None  # closures for storage

    def __repr__(self):
        return f"val: {self._val} and grad: {self._grad} and op: {self._op}"

    def __add__(self, b):
        if not isinstance(b, Value):
            b = Value(b)
        val = Value(self._val + b._val, [self, b], "+")

        def _backward():
            self._grad += 1 * val._grad
            b._grad += 1 * val._grad

        val._backward = _backward
        return val

    def __neg__(self):
        return self * (Value(-1))

    def __sub__(self, b):
        return self + (-b)

    def __truediv__(self, b):
        if not isinstance(b, Value):
            b = Value(b)
        return self * (b**-1)

    def __pow__(self, n):
        val = Value(self._val**n, [self], f"**{n}")

        def _backward():
            self._grad += n * (self._val ** (n - 1)) * val._grad

        val._backward = _backward
        return val

    def __mul__(self, b):
        if not isinstance(b, Value):
            b = Value(b)

        val = Value(self._val * b._val, [self, b], "-")

        def _backward():
            self._grad += b._val * val._grad
            b._grad += self._val * val._grad

        val._backward = _backward
        return val

    def relu(self):
        val = Value(0 if self._val < 0 else self._val, [self], "relu")

        def _backward():
            self._grad += (1 if self._val > 0 else 0) * val._grad

        val._backward = _backward
        return val

    def dropout(self, p):
        assert isinstance(p, float)
        val = Value(
            np.random.choice([self._val * (1.0 / (1.0 - p)), 0], p=[1 - p, p]),
            [self],
            "dropout",
        )

        def _backward():
            self._grad += val._grad

        val._backward = _backward
        return val

    def backward(self):
        visited = {}
        queue = []

        def _topo(graph):
            if graph in visited:
                return
            if graph._parents is not None:
                for parent in graph._parents:
                    _topo(parent)
            queue.append(graph)
            visited[graph] = True

        _topo(self)
        self._grad = 1.0
        for node in reversed(queue):
            node._backward()

    def zero_grad(self):
        self._grad = 0
