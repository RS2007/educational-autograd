import numpy as np


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

    def __pow__(self, n):
        val = Value(self._val**n, [self], f"**{n}")

        def _backward():
            self._grad += n * (self._val ** (n - 1)) * val._grad

        val._backward = _backward
        return val

    def __mul__(self, b):
        val = Value(self._val * b._val, [self, b], "-")

        def _backward():
            self._grad += b._val * val._grad
            b._grad += self._val * val._grad

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
