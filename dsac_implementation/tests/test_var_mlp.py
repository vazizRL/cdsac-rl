import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from dsac_implementation.variable_tow_headed_mlp import MLP

# Note: The first dimension is the input dimension
inp_dim = 28*28
mlp_model = MLP((inp_dim, 15, 15, 10), ('gelu', 'gelu', 'gelu'))

train_on_gpu = True
epochs = 2
batch_size = 50
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts PIL images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize the pixel values to range [-1, 1]
])
train_dataset = torchvision.datasets.MNIST(
    root='./data',  # Directory where data will be stored
    train=True,  # This is the training dataset
    transform=transform,  # Apply the defined transformations
    download=True  # Download the dataset if not already downloaded
)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,  # Number of samples in each batch
    shuffle=True  # Shuffle the data
)
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
# Define a loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(mlp_model.parameters(), lr=0.01, momentum=0.9)

# Specify device if GPU is used
device = mlp_model.device

# Train Loop
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        if train_on_gpu:
            inputs, labels = data[0].to(device), data[1].to(device)
        else:
            inputs, labels = data
        inputs = torch.flatten(inputs, start_dim=1, end_dim=-1)

        optimizer.zero_grad()

        outputs, head2 = mlp_model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        # print every
        if i % 250 == 249:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

# More detailed pred, shows information about how the classes performed
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
with torch.no_grad():
    for data in train_loader:
        if train_on_gpu:
            images, labels = data[0].to(device), data[1].to(device)
        else:
            images, labels = data
        images = torch.flatten(images, start_dim=1, end_dim=-1)
        outputs, head2 = mlp_model(images)
        _, predictions = torch.max(outputs, 1)
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class {classname:5s} is {accuracy:.1f}%')

print('Finished training')


