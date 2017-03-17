import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

class GeneratorCNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dims):
        super(Generator, self).__init__()
        self.layers = []
        
        prev_dim = input_size
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.ReLU(True))
            prev_dim = hidden_dim
        self.layers.append(nn.Linear(prev_dim, output_size))
        
        self.layer_module = ListModule(*self.layers)
        
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

class DiscriminatorCNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dims):
        super(Discriminator, self).__init__()
        self.layers = []
        
        prev_dim = input_size
        for idx, hidden_dim in enumerate(hidden_dims):
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.ReLU(True))
            prev_dim = hidden_dim
            
        self.layers.append(nn.Linear(prev_dim, output_size))
        self.layers.append(nn.Sigmoid())
        
        self.layer_module = ListModule(*self.layers)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out.view(-1, 1)

class GeneratorFC(nn.Module):
    def __init__(self, input_size, output_size, hidden_dims):
        super(Generator, self).__init__()
        self.layers = []
        
        prev_dim = input_size
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.ReLU(True))
            prev_dim = hidden_dim
        self.layers.append(nn.Linear(prev_dim, output_size))
        
        self.layer_module = ListModule(*self.layers)
        
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

class DiscriminatorFC(nn.Module):
    def __init__(self, input_size, output_size, hidden_dims):
        super(Discriminator, self).__init__()
        self.layers = []
        
        prev_dim = input_size
        for idx, hidden_dim in enumerate(hidden_dims):
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.ReLU(True))
            prev_dim = hidden_dim
            
        self.layers.append(nn.Linear(prev_dim, output_size))
        self.layers.append(nn.Sigmoid())
        
        self.layer_module = ListModule(*self.layers)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out.view(-1, 1)

class ListModule(nn.Module):
    # code from https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)
