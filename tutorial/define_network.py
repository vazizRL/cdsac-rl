import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # nn.conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
        # groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) windows
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # Flatten all dimensions except the batch dimension:
        # Syntax: torch.flatten(input, start_dim=0, end_dim=- 1)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    net = Net()
    print(f'Printed model: {net}')

    params = list(net.parameters())
    print(len(params))
    print(f'Shape of first parameter is: {params[0].shape}')

    input = torch.randn(1, 1, 32, 32)
    out = net(input)
    print(f'Output is: {out}')

    net.zero_grad()
    out.backward(torch.randn(1, 10))

    # Compute loss
    output = net(input)
    target = torch.randn(10)
    target = target.view(1, -1)
    criterion = nn.MSELoss()
    loss = criterion(output, target)
    print(f'Loss is: {loss}')

    # Display backpropagation route
    print(f'MSE Loss: {loss.grad_fn}')
    print(f'Linear  : {loss.grad_fn.next_functions[0][0]}')
    print(f'ReLU    : {loss.grad_fn.next_functions[0][0].next_functions[0][0]}')

    # Look up bias loss of conv1
    net.zero_grad()
    print(f'conv1.bias.grad before backward: {net.conv1.bias.grad}')
    loss.backward()
    print(f'conv1.bias.grad after bwackward: {net.conv1.bias.grad}')

    # Update the weights of the network manually with SGD
    learning_rate = 0.01
    for f in net.parameters():
        f.data.sub_(f.grad.data * learning_rate)

    optimizer = optim.SGD(net.parameters(), lr=0.01)
    optimizer.zero_grad()
    optimizer.step()

    


