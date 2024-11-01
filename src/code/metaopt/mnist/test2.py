import itertools
import torch
import functorch
from toolz.curried import curry, compose, identity, take_nth, accumulate, apply, map, concat, take, drop
import numpy as np
from myfunc import scan
from torch.autograd import gradcheck
import torch.nn.functional as F
from torch.func import jacrev, grad
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms



# _ = torch.manual_seed(0)

# @curry
# def predict(x, bias, weight):
#     return F.linear(x, weight, bias).sum()


# D = 2
# weight = torch.randn(D, D)
# bias = torch.randn(D)
# x = torch.randn(D)

# ft_jacobian = jacrev(predict, argnums=2)(x, bias, weight)

# print(weight)
# print(predict(x, bias, weight))
# print(ft_jacobian)
# print(grad(predict(x, bias))(weight))

# # Create two sample tensors
# tensor1 = torch.tensor([[3.0, 4.0], [1.0, 2.0]])
# tensor2 = torch.tensor([[5.0, 12.0], [0.0, 0.0]])

# # Concatenate tensors into a tuple
# tensors_tuple = (tensor1, tensor2)

# # Concatenate the tensors along dimension 0 (stacking them vertically)
# concatenated_tensor = torch.cat(tensors_tuple, dim=0)

# # Normalize the tensor along dimension 1 using L2 normalization
# normalized_tensor = torch.nn.functional.normalize(concatenated_tensor, p=2.0, dim=1)

# print(concatenated_tensor)
# print(normalized_tensor)



def datastream(numEpochs):
    return compose(itertools.chain.from_iterable, take(numEpochs), itertools.repeat)


# @curry
# def datastream(numEpochs, xs):
#     for _ in range(numEpochs):
#         for data in xs:
#             yield data


dataset = datasets.MNIST('data/mnist', train=True, download=True,
                                        transform=transforms.Compose(
                                                [transforms.ToTensor()]))
train_set, valid_set = torch.utils.data.random_split(dataset,[60000 - 100, 100])

data_loader_tr = DataLoader(train_set, batch_size=100, shuffle=True)
data_loader_vl = cycle_efficient(DataLoader(valid_set, batch_size=100, shuffle=True))

cycle_efficient = compose(itertools.chain.from_iterable, itertools.repeat)

def getEpochs(numEpochs, *dataLoaders):
    return itertools.chain.from_iterable((zip(*dataLoaders) for _ in range(numEpochs)))

# def temp(loader1, loader2):
#     return itertools.chain.from_iterable((zip(loader1, loader2) for _ in range(2)))

print(len(list(getEpochs(2, data_loader_tr))))

# xs = list(datastream(2)(zip(data_loader_tr, data_loader_vl)))
# print(len(list(datastream(2)(zip(data_loader_tr, data_loader_vl)))))

