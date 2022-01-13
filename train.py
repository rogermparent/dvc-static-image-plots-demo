from collections import defaultdict

import torch
import torch.nn as nn
import yaml
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm

from dvclive import Live

logger = Live()
torch.device("cuda" if torch.cuda.is_available() else "cpu")


def avg(nums):
    return sum(nums) / (len(nums))


def load_params():
    with open("params.yaml") as fd:
        return yaml.safe_load(fd)


def write_images(data, d):
    import matplotlib.pyplot as plt
    import os

    os.makedirs(d)
    x = data.pop("epoch")
    for key, y in data.items():
        plt.figure()
        plt.plot(x, y)
        plt.ylabel(key)
        plt.savefig(os.path.join(d, key + ".png"))


def prepare_data_loaders(batch_size, num_workers):
    train = MNIST(root="data", train=True, transform=ToTensor(), download=True,)
    test = MNIST(root="data", train=False, transform=ToTensor())
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    return train_loader, test_loader


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 8, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.out = nn.Linear(8 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


def train(model, loss_func, optimizer, num_epochs, train_loader, test_loader):
    def single_epoch():
        # train
        train_losses = []
        train_accuracies = []
        for images, labels in train_loader:
            model.train()
            b_x = Variable(images)
            b_y = Variable(labels)
            output = model(b_x)
            train_loss = loss_func(output, b_y)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_losses.append(train_loss.item())

            model.eval()
            with torch.no_grad():
                test_output = model(images)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                train_accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
                train_accuracies.append(train_accuracy)
        # test
        test_losses = []
        test_accuracies = []
        with torch.no_grad():
            for images, labels in test_loader:
                test_output = model(images)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                test_loss = loss_func(test_output, labels)
                test_losses.append(test_loss.item())
                test_accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
                test_accuracies.append(test_accuracy)

        avg_train_loss = avg(train_losses)
        avg_train_accuracy = avg(train_accuracies)
        avg_test_loss = avg(test_losses)
        avg_test_accuracy = avg(test_accuracies)
        return avg_train_loss, avg_train_accuracy, avg_test_loss, avg_test_accuracy

    def log(name, value):
        logger.log(name, value)
        data_for_images[name].append(value)

    pbar = tqdm(range(num_epochs), position=0, desc="epoch")
    for epoch in pbar:
        data_for_images["epoch"].append(epoch)

        train_loss, train_acc, test_loss, test_acc = single_epoch()

        log("train_loss", train_loss)
        log("train_accuracy", train_acc)
        log("test_loss", test_loss)
        log("test_accuracy", test_acc)
        logger.next_step()

        pbar.set_description(
            f"train loss: '{train_loss:.4f}', test_loss: '{test_loss:.4f}'"
        )


if __name__ == "__main__":
    data_for_images = defaultdict(list)

    params = load_params()
    batch_size = params["batch_size"]
    num_epochs = params["num_epochs"]
    learning_rate = params["learning_rate"]

    model = CNN()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    train_loader, test_loader = prepare_data_loaders(batch_size, 0)
    train(model, loss_func, optimizer, num_epochs, train_loader, test_loader)

    write_images(data_for_images, "plots")
