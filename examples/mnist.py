import nn
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm


# TODO: Probably incorrect code start #
def log_softmax(x):
    return x - x.exp().sum().log()


def cross_entropy_loss(x, y):
    return (-log_softmax(x) * y).sum().mean()


# TODO: Probably incorrect code ends #


class TorchMNISTNet(torch.nn.Module):
    def __init__(self):
        super(TorchMNISTNet, self).__init__()
        self.layer1 = torch.nn.Linear(28 * 28, 512)
        self.layer2 = torch.nn.Linear(512, 512)
        self.layer3 = torch.nn.Linear(512, 10)
        self.p = 0.2

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.nn.Dropout(self.p)(self.layer1(x).relu())
        x = torch.nn.Dropout(self.p)(self.layer2(x).relu())
        x = self.layer3(x)
        return x


class MyMNISTNet(nn.Module):
    def __init__(self):
        self.layer1 = nn.Linear(28 * 28, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, 10)
        self.p = 0.2

    def forward(self, x):
        x = np.reshape(x, 1 * 28 * 28)
        x = self.layer1(x).relu()
        x = self.layer2(x).relu()
        x = self.layer3(x)
        return x


def viz(image):
    plt.imshow(image, cmap="gray")
    plt.show()


def torch_mnist(train_loader, test_loader):
    model = TorchMNISTNet()
    criterion = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    epochs = 30

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


def torch_mine(test_loader, train_loader):
    model = MyMNISTNet()


if __name__ == "__main__":
    num_workers = 0
    batch_size = 20
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
    # images, labels = next(iter(train_loader))
    # viz(images.numpy()[0].reshape(28, 28))
    torch_mnist(train_loader, test_loader)
    torch_mine(train_loader, test_loader)
