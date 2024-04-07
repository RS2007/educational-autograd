#!/usr/bin/env python3.11
from tensor.tensor import Tensor

if __name__ == "__main__":
    a = Tensor([1, 2, 3, 4])
    b = Tensor([1, 2, 3, 4])
    c = a * b
    c.backward()
    print(a)
