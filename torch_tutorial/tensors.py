import torch
import numpy as np
from time import perf_counter as pf

start = pf()
# Tensor from data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# Tensor from NumPy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# From another tensor
x_ones = torch.ones_like(x_data)    # Retains prop. of x_data, overrides values
print(f'Ones Tensor: \n {x_ones} \n')
x_rand = torch.rand_like(x_data, dtype=torch.float)     # Overrides datatype of x_data, overrides values
print(f'Random Tensor: \n {x_rand} \n')

# With variety of values
shape = (2, 3)
rand_tensor = torch.rand(shape)     # Values in interval [0, 1]
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# Tensor Attributes
tensor = torch.rand(3, 4)   # Means 3x4 matrix, shape
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# Move Tensor to GPU if available
if torch.cuda.is_available():
    tensor = tensor.to('cuda')
print(f'Device tensor is stored on: {tensor.device}')


# Indexing and Slicing
tensor = torch.ones(4, 4)
tensor[:, 1] = 0
print(tensor)

# Joining tensors
t1 = torch.cat([tensor, tensor, tensor], dim=1)     # dim=1 more columns, dim=0 more rows
print(t1)

# Multiplying tensors element-wise + Alternative
print(f'tensor.mul(tensor) \n {tensor.mul(tensor)} \n')
print(f'tensor * tensor \n {tensor * tensor} \n')

# Matrix multiplication + Alternative
print(f'tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n')
print(f'tensor @ tensor.T \n {tensor @ tensor.T} \n')


# In-place operations
print(tensor, '\n')
tensor.add_(5)
print(tensor)
tensor.t_()
print(tensor)

# Memory Brdige between tensors and np.ndarray
t = torch.ones(5)
n = t.numpy()
print(f't: {t}')
print(f'n: {n}')
t.add_(1)
print(f't: {t}')
print(f'n: {n}')

n = np.ones(5)
t = torch.from_numpy(n)
np.add(n, 1, out=n)
print(f't: {t}')
print(f'n: {n}')

end = pf()

print(f'Total script run time: {end-start}')



