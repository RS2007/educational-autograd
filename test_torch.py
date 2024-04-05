import torch

x = torch.tensor([1.0], requires_grad=True)
y = torch.tensor([2.0], requires_grad=True)
f = 3 * x**5 + y
f.backward()
print(f"x.grad is {x.grad} and y.grad is {y.grad}")
