import torch
from torchvision.models import resnet18, ResNet18_Weights
from torchsummary import summary

x = torch.rand(5, 5)
y = torch.rand(5, 5)
z = torch.rand(5, 5, requires_grad=True)

a = x + y
b = x + z

print(f'Does "a" require gradients?: {a.requires_grad}')
print(f'Does "b" require gradients?: {b.requires_grad}')


# Freeze the parameters of a model
model = resnet18(weights=ResNet18_Weights.DEFAULT)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)
for param_i in model.parameters():
    param_i.requires_grad = False


model.fc = torch.nn.Linear(512, 10)





