import matplotlib.pyplot as plt
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as f
import numpy as np
from torchsummary import summary
from utils import dim_shower
from time import perf_counter


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64*6*6, 20)
        self.fc2 = nn.Linear(20, 15)
        self.fc3 = nn.Linear(15, 10)

    def forward(self, x):
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)

        return x


if __name__ == '__main__':

    train_on_gpu = True
    path = r"C:\Users\vanya\OneDrive\Desktop\HAW_Projekt\Software\PyTorch\datasets"
    batch_size = 25

    arch = (('conv', (3, 3), 64), ('pool', (2, 2)), ('conv', (3, 3), 64), ('pool', (2, 2)))
    inp_nodes = (3, 32, 32)
    dim_shower(layer_arch=arch, input_dim=(32, 32), stride=1)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=path, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=path, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Specify device if GPU is used
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('Training on GPU')
    else:
        device = torch.device('cpu')

    # Instantiate model
    net = Net()
    if train_on_gpu:
        net.to(device)

    # Define a loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    # Train network
    train_start = perf_counter()
    epochs = 2
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            if train_on_gpu:
                inputs, labels = data[0].to(device), data[1].to(device)
            else:
                inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            # print every
            if i % 250 == 249:
                print(f'[{epoch + 1}, {i+1:5d}] loss: {running_loss /2000:.3f}')
                running_loss = 0.0
    train_end = perf_counter()
    print('Finished training')

    # Test network on validation data
    correct = 0
    total = 0

    # More detailed pred, shows information about how the classes performed
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    with torch.no_grad():
        for data in testloader:
            if train_on_gpu:
                images, labels = data[0].to(device), data[1].to(device)
            else:
                images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class {classname:5s} is {accuracy:.1f}%')

    print(f'Time taken for GPU-Training:{train_on_gpu} is {train_end-train_start}')