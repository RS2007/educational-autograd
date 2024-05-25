#!/usr/bin/env python3.11
import unittest
from tensor.tensor import Value, Tensor
import torch
import nn
import numpy as np
import tinygrad


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

        with torch.no_grad():
            torch_layer = torch.nn.Linear(3, 2).eval()
            torch_layer.weight[:] = torch.tensor(layer.weights(), dtype=torch.float32)
            torch_layer.bias[:] = torch.tensor(layer.biases(), dtype=torch.float32)
            torch_x = torch.tensor([1, 2, 3], dtype=torch.float32)
            torch_res = torch.sum(torch_layer(torch_x))
        self.assertEqual(torch_res, d._val)

    def test_tensor_abstraction_add_sum(self):
        a = Tensor([1.0, 1.0], requires_grad=True)
        b = Tensor([2.0, 3.0], requires_grad=True)
        c = a + b
        d = c.sum()
        d.backward()
        a1 = tinygrad.tensor.Tensor([1.0, 1.0], requires_grad=True)
        b1 = tinygrad.tensor.Tensor([2.0, 3.0], requires_grad=True)
        c1 = a1 + b1
        d1 = c1.sum()
        d1.backward()
        self.assertEqual(a.data.tolist(), a1.numpy().tolist())
        self.assertEqual(b.data.tolist(), b1.numpy().tolist())
        self.assertEqual(a.grad.data.tolist(), a1.grad.numpy().tolist())
        self.assertEqual(b.grad.data.tolist(), b1.grad.numpy().tolist())

    def test_tensor_abstraction_mul(self):
        a = Tensor([1.0, 1.0], requires_grad=True)
        b = Tensor([2.0, 3.0], requires_grad=True)
        c = a * b
        d = c.sum()
        d.backward()
        a1 = tinygrad.tensor.Tensor([1.0, 1.0], requires_grad=True)
        b1 = tinygrad.tensor.Tensor([2.0, 3.0], requires_grad=True)
        c1 = a1 * b1
        d1 = c1.sum()
        d1.backward()
        self.assertEqual(a.data.tolist(), a1.numpy().tolist())
        self.assertEqual(b.data.tolist(), b1.numpy().tolist())
        self.assertEqual(a.grad.data.tolist(), a1.grad.numpy().tolist())
        self.assertEqual(b.grad.data.tolist(), b1.grad.numpy().tolist())

    def test_tensor_abstraction_pow(self):
        b = Tensor([2.0, 3.0], requires_grad=True)
        c = b**3
        d = c.sum()
        d.backward()
        b1 = tinygrad.tensor.Tensor([2.0, 3.0], requires_grad=True)
        c1 = b1**3
        d1 = c1.sum()
        d1.backward()
        self.assertEqual(b.data.tolist(), b1.numpy().tolist())
        self.assertEqual(b.grad.data.tolist(), b1.grad.numpy().tolist())

    def test_matrix_mul(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[5.0, 6.0], [7.0, 8.0]])
        c = a @ b
        d = c.sum()
        d.backward()
        a1 = tinygrad.tensor.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b1 = tinygrad.tensor.Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        c1 = a1 @ b1
        d1 = c1.sum()
        d1.backward()
        self.assertEqual(d.data.tolist()[0], d1.numpy().tolist())
        self.assertEqual(a.grad.data.tolist(), a1.grad.numpy().tolist())
        self.assertEqual(b.grad.data.tolist(), b1.grad.numpy().tolist())

    def test_sum_pow(self):
        aa = Tensor([[1.0, 2.0], [3.0, 4.0]])
        bb = Tensor([[5.0, 6.0], [7.0, 8.0]])
        a = aa @ bb + bb
        b = bb @ aa + aa
        c = (a.sum() + b.sum()) ** 2
        c.backward()
        aa1 = tinygrad.tensor.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        bb1 = tinygrad.tensor.Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        a1 = aa1 @ bb1 + bb1
        b1 = bb1 @ aa1 + aa1
        c1 = (a1.sum() + b1.sum()) ** 2
        c1.backward()
        self.assertEqual(c.data.tolist()[0], c1.numpy().tolist())
        self.assertEqual(a.grad.data.tolist(), a1.grad.numpy().tolist())
        self.assertEqual(b.grad.data.tolist(), b1.grad.numpy().tolist())
        self.assertEqual(aa.grad.data.tolist(), aa1.grad.numpy().tolist())
        self.assertEqual(bb.grad.data.tolist(), bb1.grad.numpy().tolist())


class TestCodegen(unittest.TestCase):
    def test_c_codegen(self):
        pass


if __name__ == "__main__":
    unittest.main()
