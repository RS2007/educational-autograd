# implementation of the optimizers
class SGD:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for index in range(len(self.parameters)):
            self.parameters[index]._val -= self.parameters[index]._grad * self.lr

    def step_tensor(self):
        for index in range(len(self.parameters)):
            self.parameters[index].data -= self.parameters[index].grad.data * self.lr

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.zero_grad()
