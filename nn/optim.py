# implementation of the optimizers
class SGD:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr
        for index in range(len(self.parameters)):
            print("In sgd constructor: ",id(self.parameters[index]))

    def step(self):
        for index in range(len(self.parameters)):
            self.parameters[index]._val -= self.parameters[index]._grad * self.lr

    def step_tensor(self):
        for index in range(len(self.parameters)):
            print("In sgd: ",id(self.parameters[index]),f" and index {index}")
            self.parameters[index].data = self.parameters[index].data - self.parameters[index].grad.data * self.lr
            print("In sgd post decrement: ",id(self.parameters[index]),f" and index {index}")

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.zero_grad()
