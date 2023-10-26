import torch
from tensordict.tensordict import TensorDict

tensordict = TensorDict({}, [])

# Set key
a = torch.rand(10)
tensordict['a'] = a

# #retrieve the value stored under 'a'
assert tensordict['a'] is a


# Like dict, return standard value if key is not found
assert tensordict.get('bananan', a) is a
# Alternatively
assert tensordict.setdefault('banana', a) is a


# Delete key + value
del tensordict['banana']


# Inplace updating of tensor, tensor is still same, values change
# Alterntively tensordict.set_('a', torch.zeros(10))
tensordict.set('a', torch.zeros(10), inplace=True)
assert (tensordict.get('a') == 0).all()
assert tensordict.get('a') is a
tensordict.set_('a', torch.ones(10))
assert (tensordict.get('a') == 1).all()
assert tensordict.get('a') is a

tensordict.rename_key_('a', 'b')
assert tensordict.get('b') is a


tensordict = TensorDict({'a': torch.rand(10), 'b': torch.rand(10)}, [10])
tensordict.update(TensorDict({'a': torch.zeros(10), 'c': torch.zeros(10)}, [10]))
assert (tensordict['a'] == 0).all()
assert (tensordict['b'] != 0).all()
assert (tensordict['c'] == 0).all()


nested_tensordict = TensorDict(
    {'a': torch.rand(2, 3), 'double_nested': {'a': torch.rand(2, 3)}}, [2, 3]
)
tensordict = TensorDict({'a': torch.rand(2), 'nested': nested_tensordict}, [2])

print(tensordict)


tensordict['nested', 'double_nested', 'b'] = torch.rand(2, 3)
tensordict.set(('nested', 'b'), torch.rand(2, 3))

print(tensordict)


# Ierating over TD Content
for key in tensordict.keys():
    print(key)

# Include nested tensordicts
for key in tensordict.keys(include_nested=True):
    print(key, '\n')

# Print only keys that are tensor values
for key in tensordict.keys(include_nested=True, leaves_only=True):
    print(key, '\n')

# Iterating over both keys and values
for key, value in tensordict.items(include_nested=True):
    if isinstance(value, TensorDict):
        print(f'{key} is a TensorDict')
    else:
        print(f'{key} is a Tensor')
print('\n', '\n')

# Look how nested values look
# print(tensordict, end='\n\n')
# print(tensordict.flatten_keys(separator='.'))

# Unflattening TensorDict
flattened_tensordict = tensordict.flatten_keys(separator='.')
print(flattened_tensordict, end='\n\n')
print(flattened_tensordict.unflatten_keys(separator='.'))










