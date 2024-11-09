from abc import ABCMeta, abstractmethod
from functools import reduce
from typing import Callable, List, TypeVar, Generic, Iterator, Union
from toolz.curried import curry, compose, map
from torch.nn import functional as f
import torch
from dataclasses import dataclass
from func import *


T = TypeVar('T')
E = TypeVar('E')
A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')
D = TypeVar('D')
X = TypeVar('X')
Y = TypeVar('Y')
H = TypeVar('H')
P = TypeVar('P')
L = TypeVar('L')
HP = TypeVar('HP')

ENV = TypeVar('ENV')

# ============== Typeclasses ==============
class HasActivation(Generic[ENV, T], metaclass=ABCMeta):
    @abstractmethod
    def getActivation(self, env: ENV) -> T:
        pass

    @abstractmethod
    def putActivation(self, s: T, env: ENV) -> ENV:
        pass


class HasLoss(Generic[ENV, T], metaclass=ABCMeta):
    @abstractmethod
    def getLoss(self, env: ENV) -> T:
        pass

    @abstractmethod
    def putLoss(self, s: T, env: ENV) -> ENV:
        pass

class HasParameter(Generic[ENV, T], metaclass=ABCMeta):
    @abstractmethod
    def getParameter(self, env: ENV) -> T:
        pass

    @abstractmethod
    def putParameter(self, s: T, env: ENV) -> ENV:
        pass

class HasHyperParameter(Generic[ENV, T], metaclass=ABCMeta):
    @abstractmethod
    def getHyperParameter(self, env: ENV) -> T:
        pass

    @abstractmethod
    def putHyperParameter(self, s: T, env: ENV) -> ENV:
        pass

# ============== Instance ==============

class OhoState(Generic[A, B, C, D]): 
    def __init__(self, activation: A, loss: B, parameter: C, hyperParameter: D):
        self.activation = activation
        self.loss = loss
        self.parameter = parameter
        self.hyperParameter = hyperParameter
    

@dataclass(frozen=True)
class VanillaRnnState(Generic[A, B, C]):
    activation: A
    loss: B
    parameter: C


class ActivationVanillaRnnStateInterpreter(HasActivation[VanillaRnnState[A, B, C], A]):
    def getActivation(self, env):
        return env.activation
    
    def putActivation(self, s, env):
        return VanillaRnnState(s, env.loss, env.parameter)
    
class LossVanillaRnnStateInterpreter(HasLoss[VanillaRnnState[A, B, C], B]):
    def getLoss(self, env):
        return env.loss
    
    def putLoss(self, s, env):
        return VanillaRnnState(env.activation, s, env.parameter)

class ParameterVanillaRnnStateInterpreter(HasParameter[VanillaRnnState[A, B, C], C]):
    def getParameter(self, env):
        return env.parameter
    
    def putParameter(self, s, env):
        return VanillaRnnState(env.activation, env.loss, s)
    
class VanillaRnnStateInterpreter(ActivationVanillaRnnStateInterpreter, LossVanillaRnnStateInterpreter, ParameterVanillaRnnStateInterpreter):
    pass

class ActivationTripletInterpreter(HasActivation[tuple[A, B, C], A]):
    def getActivation(self, env):
        a, _, _ = env 
        return a
    
    def putActivation(self, s, env):
        _, b, c = env 
        return (s, b, c)

# technique to create new typeclass instance. Just extend a previous instantiation? I know it's wierd but it's the only way to solve the expression problem. How to add new methods to existing datatypes basically. If I don't have access, how to extend functionality basically.
# See here http://ponies.io/posts/2015-07-15-solving-the-expression-problem-in-python-object-algebras-and-mypy-static-types.html
class ParameterTripletInterpreter(HasParameter[tuple[A, B, C], B]):
    def getParameter(self, env):
        _, b, _ = env 
        return b
    
    def putParameter(self, s, env):
        a, _, c = env 
        return (a, s, c)
    
class HyperParameterTripletInterpreter(HasHyperParameter[tuple[A, B, C], C]):
    def getHyperParameter(self, env):
        _, _, c = env 
        return c
    
    def putHyperParameter(self, s, env):
        a, b, _ = env 
        return (a, b, s)


class TripletInterpreter(ActivationTripletInterpreter, ParameterTripletInterpreter, HyperParameterTripletInterpreter):
    pass


class ActivationQuadrupleInterpreter(HasActivation[tuple[A, B, C, D], A]):
    def getActivation(self, env):
        a, _, _, _ = env 
        return a
    
    def putActivation(self, s, env):
        _, b, c, d = env 
        return (s, b, c, d)

class LossQuadrupleInterpreter(HasLoss[tuple[A, B, C, D], B]):
    def getLoss(self, env):
        _, b, _, _ = env 
        return b
    
    def putLoss(self, s, env):
        a, _, c, d = env 
        return (a, s, c, d)

class ParameterQuadrupleInterpreter(HasParameter[tuple[A, B, C, D], C]):
    def getParameter(self, env):
        _, _, c, _ = env 
        return c
    
    def putParameter(self, s, env):
        a, b, _, d = env 
        return (a, b, s, d)

class HyperParameterQuadrupleInterpreter(HasHyperParameter[tuple[A, B, C, D], D]):
    def getHyperParameter(self, env):
        _, _, _, d = env 
        return d
    
    def putHyperParameter(self, s, env):
        a, b, c, _ = env 
        return (a, b, c, s)

class QuadrupleInterpreter(ActivationQuadrupleInterpreter, LossQuadrupleInterpreter, ParameterQuadrupleInterpreter, HyperParameterQuadrupleInterpreter):
    pass

    
# downside is that I need to create a whole new class that inherits all the instantiations. 
# I will just maintain and reedit my god instance interpreter. Best compromise. Not truly extensible but I have full control. 

# ============== Functions ==============


def foldr(f: Callable[[A, B], B]) -> Callable[[Iterator[A], B], B]:
    def foldr_(xs: Iterator[A], x: B) -> B:
        return reduce(flip(f), xs, x)
    return foldr_


def fuse(f: Callable[[X, A], B], g: Callable[[Y, B], C]) -> Callable[[tuple[X, Y], A], C]: 
    """ g . f """
    def fuse_(pair: tuple[X, Y], a: A) -> C:
        x, y = pair
        return g(y, f(x, a))
    return fuse_

def fmapSuffix(g: Callable[[B], C], f: Callable[[X, A], B]) -> Callable[[X, A], C]:
    def fmapSuffix_(x: X, a: A) -> C:
        return g(f(x, a))
    return fmapSuffix_

def fmapPrefix(g: Callable[[A], B], f: Callable[[X, B], C]) -> Callable[[X, A], C]:
    def fmapPrefix_(x: X, a: A) -> C:
        return f(x, g(a))
    return fmapPrefix_


# ============== Expressions ==============
#!! Warning. Union is wrong. Should be intersection. Python no cap so use Union for type method inference. 
# These are a proof of concept

def offlineRnnPredict(
        t: Union[HasActivation[ENV, A], HasParameter[ENV, P]]
        , actvT: Callable[[Union[HasActivation[ENV, A], HasParameter[ENV, P]]], Callable[[X, ENV], ENV]]
        , predictT: Callable[[Union[HasParameter[ENV, P], HasActivation[A, T]]], Callable[[ENV], tuple[ENV, T]]]) -> Callable[[Iterator[X], ENV], tuple[ENV, T]]:
    actvStep = actvT(t)
    predictStep = predictT(t)
    offline = foldr(actvStep)
    return fmapSuffix(predictStep, offline)

def learnStep(t: Union[HasParameter[ENV, P], HasLoss[ENV, L]]
            , prediction: Callable[[X, ENV], tuple[ENV, T]]
            , lossT: Callable[[Union[HasLoss[ENV, L]]], Callable[[Y, tuple[ENV, T]], ENV]]
            , paramT: Callable[[Union[HasParameter[ENV, P], HasLoss[ENV, L]]], Callable[[ENV], ENV]]) -> Callable[[tuple[X, Y], ENV], ENV]:
    lossStep = lossT(t)
    paramStep = paramT(t)
    lossFn = fuse(prediction, lossStep)
    return fmapSuffix(paramStep, lossFn)

def resetRnnActivation(t: Union[HasActivation[ENV, A]]
                , resetee:  Callable[[X, ENV], ENV]
                , actv0: A) -> Callable[[X, ENV], ENV]:
    def reset_(env: ENV) -> ENV:
        return t.putActivation(actv0, env)
    return fmapPrefix(reset_, resetee)

def resetLoss(t: Union[HasLoss[ENV, L]]  # TODO: generalize this
                , resetee:  Callable[[X, ENV], ENV]
                , loss0: A) -> Callable[[X, ENV], ENV]:
    def reset_(env: ENV) -> ENV:
        return t.putLoss(loss0, env)
    return fmapPrefix(reset_, resetee)



def repeatRnnWithReset(t: Union[HasActivation[ENV, A], HasParameter[ENV, P], HasLoss[ENV, L]]
                , repeatee: Callable[[X, ENV], ENV]) -> Callable[[Iterator[X], ENV], ENV]:
    def repeat_(xs: Iterator[X], env: ENV) -> ENV:
        a0 = t.getActivation(env)
        l0 = t.getLoss(env)
        resetter = resetRnnActivation(t, repeatee, a0)
        resetter = resetLoss(t, resetter, l0)
        return foldr(resetter)(xs, env)
    return repeat_



from memory_profiler import profile
import torch 
from torch.nn import functional as f
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from typing import TypeVar, Callable, Generic, Generator, Iterator
from functools import reduce
import torchvision
import torchvision.transforms as transforms
from itertools import tee



PARAM =  tuple[torch.Tensor
            , torch.Tensor
            , torch.Tensor
            , torch.Tensor
            , torch.Tensor
            , float]

MODEL = VanillaRnnState[torch.Tensor, torch.Tensor, PARAM]


def rnnTrans(activation: Callable) -> Callable[[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor], torch.Tensor]:
    def rnnTrans_(x: torch.Tensor, param: tuple[torch.Tensor, torch.Tensor, torch.Tensor, float], h: torch.Tensor) -> torch.Tensor:
        W_in, W_rec, b_rec, alpha = param
        return (1 - alpha) * h + alpha * activation(f.linear(x, W_in, None) + f.linear(h, W_rec, b_rec))
    return rnnTrans_

def activationTrans(activationFn: Callable[[torch.Tensor], torch.Tensor]):
    def activationTrans_(t: Union[HasActivation[MODEL, torch.Tensor], HasParameter[MODEL, PARAM]]) -> Callable[[torch.Tensor, MODEL], MODEL]:
        def activationTrans__(x: torch.Tensor, env: MODEL) -> MODEL:
            a = t.getActivation(env)
            W_rec, W_in, b_rec, _, _, alpha = t.getParameter(env)
            a_ = rnnTrans(activationFn)(x, (W_in, W_rec, b_rec, alpha), a)
            return t.putActivation(a_, env)
        return activationTrans__
    return activationTrans_

def predictTrans(t: Union[HasActivation[MODEL, torch.Tensor], HasParameter[MODEL, PARAM]]) -> Callable[[MODEL], tuple[MODEL, torch.Tensor]]:
    def predictTrans_(env: MODEL) -> tuple[MODEL, torch.Tensor]:
        a = t.getActivation(env)
        _, _, _, W_out, b_out, _ = t.getParameter(env)
        return env, f.linear(a, W_out, b_out)
    return predictTrans_

def lossTrans(t: Union[HasLoss[MODEL, torch.Tensor]]) -> Callable[[torch.Tensor, tuple[MODEL, torch.Tensor]], MODEL]:
    def lossTrans_(y: torch.Tensor, pair: tuple[MODEL, torch.Tensor]) -> MODEL:
        env, prediction = pair
        loss = f.cross_entropy(prediction, y) + t.getLoss(env)
        return t.putLoss(loss, env)
    return lossTrans_

def parameterTrans(optimizer):
    def parameterTrans_(t: Union[HasParameter[MODEL, PARAM], HasLoss[MODEL, torch.Tensor]]) -> Callable[[MODEL], MODEL]:
        def parameterTrans__(env: MODEL) -> MODEL:
            optimizer.zero_grad() 
            loss = t.getLoss(env)
            loss.backward()
            optimizer.step()  # will actuall physically spooky mutate the param so no update needed. 
            print (f'Loss: {loss.item():.10f}')
            return env
        return parameterTrans__
    return parameterTrans_

def getRnn(t: VanillaRnnStateInterpreter, activationFn) -> Callable[[Iterator[torch.Tensor], MODEL], tuple[MODEL, torch.Tensor]]:
    return offlineRnnPredict(t, activationTrans(activationFn), predictTrans)

def trainRnn(t: VanillaRnnStateInterpreter
                , optimizer
                , rnn: Callable[[Iterator[torch.Tensor], MODEL], tuple[MODEL, torch.Tensor]]) -> Callable[[Iterator[tuple[Iterator[torch.Tensor], torch.Tensor]], MODEL], MODEL]:
    learn = learnStep(t, rnn, lossTrans, parameterTrans(optimizer))
    return repeatRnnWithReset(t, learn)





# Hyper-parameters 
# input_size = 784 # 28x28
num_classes = 10
num_epochs = 10
batch_size = 100

input_size = 28
sequence_length = 28
hidden_size = 128
num_layers = 2

alpha_ = 1
activation_ = f.relu
learning_rate = 0.001


# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                        train=True, 
                                        transform=transforms.ToTensor(),  
                                        download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                        train=False, 
                                        transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                        batch_size=batch_size, 
                                        shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                        batch_size=batch_size, 
                                        shuffle=False)
cleanData = map(lambda pair: (pair[0].squeeze(1).permute(1, 0, 2), pair[1])) # origin shape: [N, 1, 28, 28] -> resized: [28, N, 28]
# train_loader = cleanData(train_loader)
# test_loader = cleanData(test_loader)



W_rec_, W_in_, b_rec_, W_out_, b_out_ = initializeParametersIO(input_size, hidden_size, num_classes)
alpha_ = 1
optimizer = torch.optim.Adam([W_rec_, W_in_, b_rec_, W_out_, b_out_], lr=learning_rate)  
state0 = VanillaRnnState(torch.zeros(batch_size, hidden_size, dtype=torch.float32)
                        , 0
                        , (W_rec_, W_in_, b_rec_, W_out_, b_out_, alpha_))

rnn = getRnn(VanillaRnnStateInterpreter(), activation_)
trainFn = trainRnn(VanillaRnnStateInterpreter(), optimizer, rnn)
trainEpochsFn = repeatRnnWithReset(VanillaRnnStateInterpreter(), trainFn)

epochs = [cleanData(train_loader) for _ in range(num_epochs)]
stateTrained = trainEpochsFn(epochs, state0)




# def predict(output, target):
#     _, predicted = torch.max(output.data, 1)
#     n_samples = target.size(0)
#     n_correct = (predicted == target).sum().item()
#     return (n_samples, n_correct)


# accuracy = compose(   lambda pair: 100.0 * pair[1] / pair[0]
#                     , totalStatistic(predict, lambda res, pair: (res[0] + pair[0], res[1] + pair[1])))

# with torch.no_grad():
#     xs_test, ys_test = tee(test_loader, 2)
#     xtream_test, targets_test = map(compose(lambda x: (x, None), fst), xs_test), map(snd, ys_test)

#     def getReadout(pair):
#         h, (_, rd) = pair 
#         return rd(h)
#     testOuputs = compose( map(getReadout)
#                         , getOutputs((-1, h0, (pN, readoutN))))
#     print(accuracy(testOuputs, xtream_test, targets_test))


