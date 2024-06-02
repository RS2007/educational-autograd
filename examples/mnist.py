import nn
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensor.tensor import Tensor
from nn.optim import SGD
import matplotlib.pyplot as plt


def log_softmax(x):
    y = x - x.max()
    return y - y.exp().sum().log()


def cross_entropy_loss(x, y):
    return (-log_softmax(x) * y).sum()

class TorchMNISTNet(torch.nn.Module):
    def __init__(self):
        super(TorchMNISTNet, self).__init__()
        self.layer1 = torch.nn.Linear(28 * 28, 512)
        self.layer2 = torch.nn.Linear(512, 512)
        self.layer3 = torch.nn.Linear(512, 10)
        self.p = 0.2

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = (self.layer1(x).relu())
        x = (self.layer2(x).relu())
        x = self.layer3(x)
        return x


class MyMNISTNet(nn.Module):
    def __init__(self):
        self.layer1 = nn.Linear2(28 * 28, 512)
        self.layer2 = nn.Linear2(512, 512)
        self.layer3 = nn.Linear2(512, 10)

    def forward(self, x):
        x.data = x.data.reshape(-1,28*28)
        x = self.layer1(x).relu()
        #print("After layer 1: ",x.data)
        x = self.layer2(x).relu()
        x = self.layer3(x)
        return x
    def parameters(self):
        return np.hstack([self.layer1.parameters(),self.layer2.parameters(),self.layer3.parameters()])



def viz(image):
    plt.imshow(image, cmap="gray")
    plt.show()


def torch_mnist(train_loader, test_loader):
    model = TorchMNISTNet()
    print("First layer from torch: ",model.layer1.weight)
    criterion = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=0.0005)
    epochs = 2

    for i in tqdm(range(epochs)):
        train_loss = 0.0

        for data, target in train_loader:
            optim.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optim.step()
            train_loss += loss.item() * data.size(0)

        train_loss = train_loss / len(train_loader.dataset)

        print("Epoch: {} \tTraining Loss: {:.6f}".format(i + 1, train_loss))

    # TODO: save the mlp weights to
    torch.save(model.state_dict(), "./mnist-weights.pth")
    test_loss = 0.0
    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))
    model.eval()
    for data, target in test_loader:
        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item() * data.size(0)
        _, pred = torch.max(output, 1)
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        for i in range(batch_size):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    test_loss = test_loss / len(test_loader.dataset)
    print("Test Loss: {:.6f}\n".format(test_loss))
    for i in range(10):
        if class_total[i] > 0:
            print(
                "Test Accuracy of %5s: %2d%% (%2d/%2d)"
                % (
                    str(i),
                    100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]),
                    np.sum(class_total[i]),
                )
            )

    print(
        "\nTest Accuracy (Overall): %2d%% (%2d/%2d)"
        % (
            100.0 * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct),
            np.sum(class_total),
        )
    )


def get_randomized_batch(inputs,results,batch_size):
    randomized_indices = list(range(0, len(inputs), batch_size))
    np.random.shuffle(randomized_indices)
    for i in randomized_indices:
        yield inputs[i : i + batch_size], results[i : i + batch_size]


def torch_mine(test_loader, train_loader):
    model = MyMNISTNet()
    epochs = 20
    optim = SGD(model.parameters(),lr=0.01)
    losses = np.array([0 for _ in range(epochs)])

    for i in tqdm(range(epochs)):
        train_loss = Tensor([0.0])
        inputs,results = train_loader.data.numpy(),train_loader.targets.numpy()
        dataiter = get_randomized_batch(inputs,results,32)
        for indx,(data, target) in enumerate(dataiter):
            data = Tensor(data.tolist())
            target = Tensor(target.tolist())
            optim.zero_grad()
            output = model(data)
            one_hot_target = np.array([np.eye(10)[int(target_elem)-1] for target_elem in target.data])
            loss = cross_entropy_loss(output,Tensor(one_hot_target)).sum() / output.data.shape[0]
            loss.backward()
            optim.step_tensor()
            train_loss += loss.sum().data[0]

        train_loss = train_loss / inputs.shape[0]
        losses[i] = train_loss.data[0]

        print(f"Epoch: {i} \tTraining Loss: {train_loss.data}")

    plt.plot(range(1,epochs+1),losses)
    plt.show()

    test_inputs,test_results = train_loader.data.numpy(),train_loader.targets.numpy()
    true_cases = 0
    for (input,output) in zip(test_inputs,test_results):
        input = Tensor(input.tolist())
        output = Tensor(output.tolist())
        model_out = model(input)
        if (np.argmax(model_out.data)+1) == output.data[0]:
            true_cases += 1
    print(f"Accuracy: {true_cases/len(test_inputs)}")






if __name__ == "__main__":
    num_workers = 0
    batch_size = 1
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        root="data", train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, num_workers=num_workers
    )
    images, labels = next(iter(train_loader))
    print(labels[0])

    # viz(images.numpy()[0].reshape(28, 28))
    #torch_mnist(train_loader, test_loader)
    torch_mine(train_data, test_data)
