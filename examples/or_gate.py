#!/usr/bin/env python3.11
import nn
from nn.optim import SGD
from tensor.tensor import Value
import numpy as np
import torch


def mine(inputs, outputs):
    layer1 = nn.Linear(2, 2)
    layer2 = nn.Linear(2, 1)
    optim = SGD(np.hstack([layer1.parameters(), layer2.parameters()]), 0.001)
    for i in range(10000):
        loss = Value(0)
        for [x1, y1], output in zip(inputs, outputs):
            net_out = layer2(layer1([Value(x1), Value(y1)]))[0].relu().dropout(0.4)
            loss += (net_out - output) ** 2
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(f"Loss is {loss}")
    print("Results")
    for [x1, y1], output in zip(inputs, outputs):
        net_out = layer2(layer1([Value(x1), Value(y1)]))[0].relu()
        print(
            f"For {x1} and {y1}, model predicted {round(net_out._val,0)} , actual: {output}"
        )


def pytorch(inputs, outputs):
    layer1 = torch.nn.Linear(2, 2)
    layer2 = torch.nn.Linear(2, 1)
    layer3 = torch.nn.Dropout(0.4)
    model = torch.nn.Sequential(layer1, layer2, layer3)
    optim = torch.optim.SGD(model.parameters(), 0.001)
    for i in range(10000):
        loss = 0
        for [x1, y1], output in zip(inputs, outputs):
            net_out = model(torch.Tensor([x1, y1])).relu()
            loss += (net_out - output) ** 2
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(f"Loss is {loss}")
    print("Results")
    for [x1, y1], output in zip(inputs, outputs):
        net_out = model(torch.Tensor([x1, y1])).relu()
        print(
            f"For {x1} and {y1}, model predicted {1 if net_out >= 0.5 else 0} , actual: {output}"
        )


if __name__ == "__main__":
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    outputs = np.array([0, 1, 1, 1])
    mine(inputs, outputs)
    pytorch(inputs, outputs)
