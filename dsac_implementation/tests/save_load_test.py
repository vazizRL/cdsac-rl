import torch
import torch as T
import os
from dsac_implementation.networks import Actor

device = torch.device('cuda:0')
actor = Actor(1, 2, (2, 2), ('gelu', 'gelu', 'gelu'), -2, 2, -1, 1)
rnd_tensor = T.tensor((2,), dtype=torch.float32).to(device)

output = actor(rnd_tensor)
print(f'Before saving - Input: {rnd_tensor.item()}; Output: {output}')

curr_dir = os.getcwd() + '/'
path = curr_dir + 'model_tensor.tar'

torch.save({
    'input': rnd_tensor,
    'actor_state_dict': actor.state_dict()},
    f=path)

# Initialize new actor
actor_loaded = Actor(1, 2, (2, 2), ('gelu', 'gelu', 'gelu'), -2, 2, -1, 1)
checkpoint = torch.load(path)
actor_loaded.load_state_dict(checkpoint['actor_state_dict'])
input_loaded = checkpoint['input']

output_new = actor_loaded(input_loaded)
print(f'After saving - Input: {input_loaded.item()}, Output: {output_new}')
