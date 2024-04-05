#!/usr/bin/env python3.11
import unittest
from tensor.tensor import Value
import torch
import nn
import numpy as np


class TestValBasic(unittest.TestCase):
    def test_basic_ops(self):
        val1 = Value(2.0)
        val2 = Value(3.0)
        sq = (val2 - val1) ** 2
        self.assertEqual(sq._val, 1.0)
        sq.backward()
        self.assertEqual(val1._grad, -2.0)
        self.assertEqual(val2._grad, 2.0)

    def test_with_torch(self):
        x = torch.tensor([1.0], requires_grad=True)
        y = torch.tensor([2.0], requires_grad=True)
        f = 3 * x**5 + y
        f.backward()
        x_grad_torch, y_grad_torch = x.grad, y.grad
        x = Value(1.0)
        y = Value(2.0)
        f = (x**5) * 3 + y
        f.backward()
        x_grad, y_grad = x._grad, y._grad
        self.assertEqual(x_grad_torch, x_grad)
        self.assertEqual(y_grad_torch, y_grad)

    def test_next(self):
        a = Value(1)
        b = Value(2)
        c = Value(3)
        layer = nn.Linear(3, 2)
        d = sum([x for x in layer(np.array([a, b, c]))], Value(0))
        d.backward()
        print(a)

        with torch.no_grad():
            torch_layer = torch.nn.Linear(3, 2).eval()
            torch_layer.weight[:] = torch.tensor(layer.weights(), dtype=torch.float32)
            torch_layer.bias[:] = torch.tensor(layer.biases(), dtype=torch.float32)
            torch_x = torch.tensor([1, 2, 3], dtype=torch.float32)
            torch_res = torch.sum(torch_layer(torch_x))
        self.assertEqual(torch_res, d._val)


if __name__ == "__main__":
    unittest.main()
