import torch
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.DEFAULT)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

# Forward pass
prediction = model(data)

# Calc. loss and gradients
loss = (prediction - labels).sum()
loss.backward()

# Load optimizer,
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# Execute a step in the descent direction
optim.step()

