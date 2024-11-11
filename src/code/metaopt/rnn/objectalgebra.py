from abc import ABCMeta, abstractmethod
from functools import reduce
from typing import Callable, List, TypeVar, Generic, Iterator, Union
from toolz.curried import curry, compose, map
from torch.nn import functional as f
import torch
from dataclasses import dataclass
from func import *
from matplotlib import pyplot as plt
from torchviz import make_dot
from operator import add


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
