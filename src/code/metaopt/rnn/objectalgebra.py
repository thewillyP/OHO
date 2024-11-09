from abc import ABCMeta, abstractmethod
from functools import reduce
from typing import Callable, List, TypeVar, Generic, Iterator, Union
from toolz.curried import curry, compose
from torch.nn import functional as f
import torch
from dataclasses import dataclass
from func import *


"""
1. I don't have diamond inheritance issues because I never override functions. 
2. I will never create a diamond bc my use case only has linear. 
Yes I will have to implement an instance for (a, b) different from (a,b,c) but it is
a) located in one place so more manageable
b) Maybe intrinsic. WHat is the canonical way to get and set different tuples that are extended on each other? That is unclear. 

This buys me generality. I just have to code one function for my update guys and that's it. All I need after is to fold them and boom. I'm done. 
oho is literally the dream function.
I'm basically doing dependency injection. 

Oh what if I want to reuse a guy I created? That's no longer a valid question anymore because all I need to do is pass in a different environment. 

"""

T = TypeVar('T')
E = TypeVar('E')
A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')
D = TypeVar('D')
X = TypeVar('X')
Y = TypeVar('Y')
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


# class IsMonoid(Generic[T], metaclass=ABCMeta):
#     @abstractmethod
#     def mempty(self) -> T:
#         pass

#     @abstractmethod
#     def mappend(self, x: T, y: T) -> T:
#         pass

# class MonoidIntInterpreter(IsMonoid[int]):
#     def mempty(self) -> int:
#         return 0

#     def mappend(self, x: int, y: int) -> int:
#         return x + y

    
# downside is that I need to create a whole new class that inherits all the instantiations. 
# I will just maintain and reedit my god instance interpreter. Best compromise. Not truly extensible but I have full control. 

# ============== Functions ==============

"""
3 patterns
1) fold 
2) fuse
3) append

"""

def foldr(f: Callable[[A, B], B]) -> Callable[[Iterator[A], B], B]:
    def foldr_(xs: Iterator[A], x: B) -> B:
        return reduce(flip(f), xs, x)
    return foldr_


def fuse(f: Callable[[A, B], B], g: Callable[[C, B], B]) -> Callable[[tuple[A, C], B], B]: 
    """ f . g """
    def fuse_(pair: tuple[A, C], x: B) -> B:
        a, c = pair
        return f(a, g(c, x))
    return fuse_

def fuseSnd(f: Callable[[A, C], C], g: Callable[[C], D], h: Callable[[D, C], C]) -> Callable[[A, C], C]:
    """ composeSnd(offline, liftA2(apply, curry(flip(paramTrans)), t.getActivation)) """
    def fuseSnd_(a: A, c: C) -> C:
        c_ = f(a, c)
        return h(g(c_), c_)
    return fuseSnd_

# ============== Expressions ==============
#!! Warning. Union is wrong. Should be intersection. Python no cap so use Union for type method inference. 
# These are a proof of concept


def stepLiteral0(param1: Callable[[ENV], T], updateEnv: Callable[[A, ENV], ENV], stepFn: Callable[[X, T], A]) -> Callable[[X, ENV], ENV]:
    def stepLiteral0_(x: X, env: ENV) -> ENV:
        t = param1(env)
        b = stepFn(x, t)
        return updateEnv(b, env)
    return stepLiteral0_

def stepLiteral1(param1: Callable[[ENV], T], param2: Callable[[ENV], E], updateEnv: Callable[[A, ENV], ENV], stepFn: Callable[[X, T, E], A]) -> Callable[[X, ENV], ENV]:
    def stepLiteral_(x: X, env: ENV) -> ENV:
        t = param1(env)
        e = param2(env)
        t_ = stepFn(x, t, e)
        return updateEnv(t_, env)
    return stepLiteral_

def stepLiteral2(param1: Callable[[ENV], T], param2: Callable[[ENV], E], param3: Callable[[ENV], A], updateEnv: Callable[[B, ENV], ENV], stepFn: Callable[[X, T, E, A], B]) -> Callable[[X, ENV], ENV]:
    def stepLiteral2_(x: X, env: ENV) -> ENV:
        t = param1(env)
        e = param2(env)
        a = param3(env)
        b = stepFn(x, t, e, a)
        return updateEnv(b, env)
    return stepLiteral2_




def rnnActivationStep(t: Union[HasActivation[ENV, T], HasParameter[ENV, E]], rnnT: Callable[[X, E, T], T]) -> Callable[[X, ENV], ENV]:
    return stepLiteral1(t.getActivation, t.getParameter, t.putActivation, rnnT)

# paramStep(t, ffwdParamStep) = feed forward parameter update
# paramStep(t, rnnParamStep) = rnn parameter update. Takes hidden state this time. 
def paramStep(t: Union[HasParameter[ENV, E]], paramT: Callable[[X, E], E]) -> Callable[[X, ENV], ENV]:
    def paramStep_(x: X, env: ENV) -> ENV:
        p = t.getParameter(env)
        p_ = paramT(x, p)
        return t.putParameter(p_, env)
    return paramStep_


def metaStep(t: Union[HasParameter[ENV, T], HasHyperParameter[ENV, E]], ohoT: Callable[[X, T, E], E]) -> Callable[[X, ENV], ENV]:
    return stepLiteral1(t.getParameter, t.getHyperParameter, t.putHyperParameter, ohoT)


def onlineRnnStep(t: Union[HasActivation[ENV, T], HasParameter[ENV, E]], rnnT: Callable[[X, E, T], T], paramT: Callable[[T, E], E]) -> Callable[[X, ENV], ENV]:
    rnnTrans = rnnActivationStep(t, rnnT)
    paramTrans = paramStep(t, paramT)
    return fuseSnd(rnnTrans, t.getActivation, paramTrans)
    # def onlineRnn_(x: X, env: ENV) -> ENV:  # not using point free style means prone to errors like mixing up env, env_ but not pythonic + less type checking so whatevs
    #     env_ = rnnTrans(x, env)
    #     a = t.getActivation(env_)
    #     return paramTrans(a, env_)
    # return onlineRnn_

def onlineRnn(t: Union[HasActivation[ENV, T], HasParameter[ENV, E]], rnnT: Callable[[X, E, T], T], paramT: Callable[[T, E], E]) -> Callable[[Iterator[X], ENV], ENV]:
    rnnTrans = onlineRnnStep(t, rnnT, paramT)
    return foldr(rnnTrans)

def offlineRnnActivation(t: Union[HasActivation[ENV, T]], rnnT: Callable[[X, E, T], T]) -> Callable[[Iterator[X], ENV], ENV]:
    # rnnTrans = onlineRnnStep(t, rnnT, lambda _, p: p)  # shows equivalence to onlineRnn. Want to get rid of parameter update capability though so not use.
    rnnTrans = rnnActivationStep(t, rnnT)
    return foldr(rnnTrans)

def offlineRnn(t: Union[HasActivation[ENV, T], HasParameter[ENV, E]], rnnT: Callable[[X, E, T], T], paramT: Callable[[T, E], E]) -> Callable[[Iterator[X], ENV], ENV]:
    offline = offlineRnnActivation(t, rnnT)
    paramTrans = paramStep(t, paramT)
    return fuseSnd(offline, t.getActivation, paramTrans)

# what's a good name for doing online first with paramT1, then offline with paramT2?
def onlineThenOffline(t: Union[HasActivation[ENV, T], HasParameter[ENV, E]], rnnT: Callable[[X, E, T], T], paramT1: Callable[[T, E], E], paramT2: Callable[[T, E], E]) -> Callable[[Iterator[X], ENV], ENV]:
    offline = onlineRnn(t, rnnT, paramT1)
    paramTrans = paramStep(t, paramT2)
    return fuseSnd(offline, t.getActivation, paramTrans)

def offlineRnnOho(t: Union[HasActivation[ENV, T], HasParameter[ENV, E], HasHyperParameter[ENV, A]], rnnT: Callable[[X, E, T], T], paramT: Callable[[T, E], E], ohoT: Callable[[Y, T, A], A]) -> Callable[[tuple[Iterator[X], Y], ENV], ENV]:
    offline = offlineRnn(t, rnnT, paramT)
    ohoTrans = metaStep(t, ohoT)
    return fuse(offline, ohoTrans)

def onlineMetaRnnStep(t: Union[HasActivation[ENV, T], HasParameter[ENV, E], HasHyperParameter[ENV, A]], rnnT: Callable[[X, E, T], T], paramT: Callable[[T, E], E], ohoT: Callable[[Y, T, A], A]) -> Callable[[tuple[X, Y], ENV], ENV]:
    online = onlineRnnStep(t, rnnT, paramT)
    ohoTrans = metaStep(t, ohoT)
    return fuse(online, ohoTrans)

def onlineMetaRnn(t: Union[HasActivation[ENV, T], HasParameter[ENV, E], HasHyperParameter[ENV, A]], rnnT: Callable[[X, E, T], T], paramT: Callable[[T, E], E], ohoT: Callable[[Y, T, A], A]) -> Callable[[Iterator[tuple[X, Y]], ENV], ENV]:
    online = onlineMetaRnnStep(t, rnnT, paramT, ohoT)
    return foldr(online)


def rnnLossSeq(t: Union[HasActivation[ENV, T], HasParameter[ENV, E], HasLoss[ENV, B]], actvT: Callable[[X, E, T], T], lossT: Callable[[Y, E, T, B], B]) -> Callable[[tuple[X, Y], ENV], ENV]:
    actv = stepLiteral1(t.getActivation, t.getParameter, t.putActivation, actvT)
    def lossStep_(y: Y, env: ENV) -> ENV:
        a = t.getActivation(env)
        p = t.getParameter(env)
        l = t.getLoss(env)
        b_ = lossT(y, p, a, l)
        return t.putLoss(b_, env)
    return fuse(actv, lossStep_)

# I should make these combinations in line, since there's a combinatorial explosion of all posible combinations. I have the pattern, just need to assemble on demand. 

# If I need to add oho to offline then online,
"""
TODO
0) fix union types to be interesection types - python no union types
1) offline rnn - ✅
2) offline rnn + oho - ✅
3) online rnn + oho - ✅
4) feedforward + oho
pytorch implementation should just be 
a) pass jacobian or something
b) pass model

rnnParam step
fuse two ways, 
update after offline. this is get the backprop graph and call autograd on it.
collect info during offline, then update after. This does require me fusing some info during offlineRnn, so not true offline. 
let's make online rnn first, then see if the above is a special case
"""






# activationT = rnnT(f.relu)
# test2 = rnnActivationStep(TripletInterpreter(), test)



from memory_profiler import profile
import torch 
from torch.nn import functional as f
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from typing import TypeVar, Callable, Generic, Generator, Iterator
from toolz.curried import curry, compose, identity, take_nth, accumulate, apply, map, concat, take, drop, mapcat, last
from functools import reduce
import torchvision
import torchvision.transforms as transforms
from itertools import tee



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

def activationTrans(t: Union[HasActivation[MODEL, torch.Tensor], HasParameter[MODEL, PARAM]]) -> Callable[[torch.Tensor, MODEL], MODEL]:
    def activationTrans_(x: torch.Tensor, env: MODEL) -> MODEL:
        a = t.getActivation(env)
        W_rec, W_in, b_rec, _, _, alpha = t.getParameter(env)
        a_ = rnnTrans(f.relu)(x, (W_in, W_rec, b_rec, alpha), a)
        return t.putActivation(a_, env)
    return activationTrans_

def predictTrans(t: Union[HasActivation[MODEL, torch.Tensor], HasParameter[MODEL, PARAM]]) -> Callable[[MODEL], tuple[MODEL, torch.Tensor]]:
    def predictTrans_(env: MODEL) -> torch.Tensor:
        a = t.getActivation(env)
        _, _, _, W_out, b_out, _ = t.getParameter(env)
        return env, f.linear(a, W_out, b_out)
    return predictTrans_

def lossTrans(t: Union[HasLoss[MODEL, torch.Tensor]]) -> Callable[[Y, tuple[MODEL, torch.Tensor]], MODEL]:
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



W_rec_, W_in_, b_rec_, W_out_, b_out_ = initializeParametersIO(input_size, hidden_size, num_classes)
alpha_ = 1
optimizer = torch.optim.Adam([W_rec_, W_in_, b_rec_, W_out_, b_out_], lr=learning_rate)  
state0 = VanillaRnnState(torch.zeros(batch_size, hidden_size, dtype=torch.float32)
                        , 0
                        , (W_rec_, W_in_, b_rec_, W_out_, b_out_, alpha_))


def classifyImage(
        t: Union[HasActivation[ENV, T], HasParameter[ENV, E]]
        , actvT: Callable[[Union[HasActivation[ENV, T], HasParameter[ENV, E]]], Callable[[X, ENV], ENV]]
        , predictT: Callable[[Union[HasParameter[ENV, E], HasActivation[ENV, T]]], Callable[[ENV,  tuple[ENV, A]]]]) -> Callable[[Iterator[X], ENV], tuple[ENV, A]]:
    actvStep = actvT(t)
    predictStep = predictT(t)
    offline = foldr(actvStep)
    def predict_(xs: Iterator[X], env: ENV) -> tuple[ENV, A]: # fmap (predictStep .) offline
        env_ = offline(xs, env)
        return predictStep(env_)
    return predict_



# def classifyImage(t: Union[HasActivation[ENV, T], HasParameter[ENV, E]], actvT: Callable[[X, E, T], T], predictT: Callable[[E, T], B]) -> Callable[[Iterator[X], ENV], B]:
#     predictedImage = offlineRnnActivation(t, actvT)
#     def predictFn(xs: Iterator[X], env: ENV) -> B:
#         env_ = predictedImage(xs, env)
#         return predictT(t.getParameter(env_), t.getActivation(env_))
#     return predictFn

def learnImage(t: Union[HasActivation[ENV, T], HasParameter[ENV, E], HasLoss[ENV, B]], actvT: Callable[[X, E, T], T], predictT: Callable[[E, T], C], lossT: Callable[[Y, C, B], B], paramT: Callable[[B, E], E]) -> Callable[[tuple[Iterator[X], Y], ENV], ENV]:
    predictedImage = offlineRnnActivation(t, actvT)
    lossFn = stepLiteral2(t.getActivation, t.getParameter, t.getLoss, t.putLoss, lossT)
    predictedLoss = fuse(predictedImage, lossFn)
    paramTrans = stepLiteral0(t.getParameter, t.putParameter, paramT)
    learnStep = fuseSnd(predictedLoss, t.getLoss, paramTrans)
    return learnStep


def learnImage(t: Union[HasActivation[ENV, T], HasParameter[ENV, E], HasLoss[ENV, B]], actvT: Callable[[X, E, T], T], lossT: Callable[[Y, E, T, B], B], paramT: Callable[[B, E], E]) -> Callable[[Iterator[tuple[Iterator[X], Y]], ENV], ENV]:
    learnStep = learnImage(t, actvT, lossT, paramT)
    def learn(xs: Iterator[tuple[Iterator[X], Y]], _env: ENV) -> ENV:
        a0 = t.getActivation(_env)
        resetter: Callable[[ENV], ENV] = lambda env: t.putActivation(a0, env)      
        resetActivation = fmap(lambda g: compose2(resetter, g), lambda x: lambda env: learnStep(x, env))  # fmap (reset .) learnStep
        return foldr(lambda pair, env: resetActivation(pair)(env))(xs, _env)
    return learn

# also need to have a prediction only model




def predict(output, target):
    _, predicted = torch.max(output.data, 1)
    n_samples = target.size(0)
    n_correct = (predicted == target).sum().item()
    return (n_samples, n_correct)


accuracy = compose(   lambda pair: 100.0 * pair[1] / pair[0]
                    , totalStatistic(predict, lambda res, pair: (res[0] + pair[0], res[1] + pair[1])))

with torch.no_grad():
    xs_test, ys_test = tee(test_loader, 2)
    xtream_test, targets_test = map(compose(lambda x: (x, None), fst), xs_test), map(snd, ys_test)

    def getReadout(pair):
        h, (_, rd) = pair 
        return rd(h)
    testOuputs = compose( map(getReadout)
                        , getOutputs((-1, h0, (pN, readoutN))))
    print(accuracy(testOuputs, xtream_test, targets_test))



t1: float = 6
t2: float = 4
a: float = 2
b: float = -1
t1_dur: float = 0.99
t2_dur: float = 0.99
outT: float = 10
st, et = 0., 11.
addMemoryTask: Callable[[float], tuple[float, float, float]] = createAddMemoryTask(t1, t2, a, b, t1_dur, t2_dur, outT)

