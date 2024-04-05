#!/usr/bin/env python3
from tensor.tensor import Value


if __name__ == "__main__":
    val1 = Value(2)
    val2 = Value(3)
    sq = (val2 - val1) ** 1
    print("Before")
    print(val1)
    print(val2)
    print(sq)
    sq.backward()
    print("After")
    print(val1)
    print(val2)
    print(sq)
