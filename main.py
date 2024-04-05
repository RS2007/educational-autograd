#!/usr/bin/env python3
import nn
from nn.optim import SGD
from tensor.tensor import Value
import numpy as np


if __name__ == "__main__":
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    outputs = np.array([0, 1, 1, 1])
    a = Value(1)
    b = Value(2)
    c = Value(3)
    layer1 = nn.Linear(2, 2)
    layer2 = nn.Linear(2, 1)
    optim = SGD(np.hstack([layer1.parameters(), layer2.parameters()]), 0.01)
    for i in range(10000):
        loss = Value(0)
        for [x1, y1], output in zip(inputs, outputs):
            net_out = layer2(layer1([Value(x1), Value(y1)]))
            loss += (net_out[0] - output) ** 2
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(f"Loss is {loss}")
    print("Results")
    for [x1, y1], output in zip(inputs, outputs):
        net_out = layer2(layer1([Value(x1), Value(y1)]))
        print(
            f"For {x1} and {y1}, model predicted {round(net_out[0]._val,0)} , actual: {output}"
        )
