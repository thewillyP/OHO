from abc import ABCMeta, abstractmethod
from functools import reduce
from typing import Callable, TypeVar, Generic, Union, Iterator
from toolz.curried import curry, compose
from torch.nn import functional as f
import torch


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
X = TypeVar('X')
Y = TypeVar('Y')
ENV = TypeVar('ENV')


class HasActivation(Generic[ENV, T], metaclass=ABCMeta):
    @abstractmethod
    def getActivation(self, env: ENV) -> T:
        pass

    @abstractmethod
    def putActivation(self, s: T, env: ENV) -> E:
        pass


class HasParameter(Generic[ENV, T], metaclass=ABCMeta):
    @abstractmethod
    def getParameter(self, env: ENV) -> T:
        pass

    @abstractmethod
    def putParameter(self, s: T, env: ENV) -> E:
        pass

class HasHyperParameter(Generic[ENV, T], metaclass=ABCMeta):
    @abstractmethod
    def getHyperParameter(self, env: ENV) -> T:
        pass

    @abstractmethod
    def putHyperParameter(self, s: T, env: ENV) -> E:
        pass

class OhoState(HasActivation[tuple[A, B, C], A], HasParameter[tuple[A, B, C], B], HasHyperParameter[tuple[A, B, C], C]):

    def getActivation(self, env):
        a, _, _ = env 
        return a
    
    def putActivation(self, s, env):
        _, b, c = env 
        return (s, b, c)
    
    def getParameter(self, env):
        _, b, _ = env 
        return b
    
    def putParameter(self, s, env):
        a, _, c = env 
        return (a, s, c)
    
    def getHyperParameter(self, env):
        _, _, c = env 
        return c
    
    def putHyperParameter(self, s, env):
        a, b, _ = env 
        return (a, b, s)

"""
-- updateA :: (HasA env) => a -> env -> env
-- updateA _ = liftA2 ($) putA getA
"""


def foldr(f: Callable[[A, B], B]) -> Callable[[Iterator[A], B], B]:
    def foldr_(xs: Iterator[A], x: B) -> B:
        return reduce(f, xs, x)
    return foldr_


def fuse(f: Callable[[A, B], B], g: Callable[[C, B], B]) -> Callable[[tuple[A, C], B], B]: 
    """ f . g """
    def fuse_(pair: tuple[A, C], x: B) -> B:
        a, c = pair
        return f(a, g(c, x))
    return fuse_



def rnnActivationStep(t: Union[HasActivation[ENV, T], HasParameter[ENV, E]], rnnT: Callable[[X, E, T], T]) -> Callable[[X, ENV], ENV]:
    def rnnActivationStep_(x: X, env: ENV) -> ENV:
        a = t.getActivation(env)
        p = t.getParameter(env)
        a_ = rnnT(x, p, a)
        return t.putActivation(a_, env)
    return rnnActivationStep_


# paramStep(t, ffwdParamStep) = feed forward parameter update
# paramStep(t, rnnParamStep) = rnn parameter update. Takes hidden state this time. 
def paramStep(t: Union[HasParameter[ENV, E]], paramT: Callable[[X, E], E]) -> Callable[[X, ENV], ENV]:
    def paramStep_(x: X, env: ENV) -> ENV:
        p = t.getParameter(env)
        p_ = paramT(x, p)
        return t.putParameter(p_, env)
    return paramStep_


def onlineRnnStep(t: Union[HasActivation[ENV, T], HasParameter[ENV, E]], rnnT: Callable[[X, E, T], T], paramT: Callable[[T, E], E]) -> Callable[[X, ENV], ENV]:
    rnnTrans = rnnActivationStep(t, rnnT)
    paramTrans = paramStep(t, paramT)
    def onlineRnn_(x: X, env: ENV) -> ENV:  # not using point free style means prone to errors like mixing up env, env_ but not pythonic + less type checking so whatevs
        env_ = rnnTrans(x, env)
        a = t.getActivation(env_)
        return paramTrans(a, env_)
    return onlineRnn_

def onlineRnn(t: Union[HasActivation[ENV, T], HasParameter[ENV, E]], rnnT: Callable[[X, E, T], T], paramT: Callable[[T, E], E]) -> Callable[[Iterator[X], ENV], ENV]:
    rnnTrans = onlineRnnStep(t, rnnT, paramT)
    return foldr(rnnTrans)

def offlineRnn(t: Union[HasActivation[ENV, T], HasParameter[ENV, E]], rnnT: Callable[[X, E, T], T]) -> Callable[[Iterator[X], ENV], ENV]:
    rnnTrans = onlineRnnStep(t, rnnT, lambda _, p: p)
    return foldr(rnnTrans)


# def offlineRnn(t: Union[HasActivation[ENV, T], HasParameter[ENV, E]], rnnT: Callable[[X, E, T], T]) -> Callable[[Iterator[X], ENV], ENV]:
#     rnnTrans = rnnActivationStep(t, rnnT)
#     return foldr(rnnTrans)


# def offlineRnnUpdate(t: Union[HasActivation[ENV, T], HasParameter[ENV, E]], rnnT: Callable[[X, E, T], T], paramT: Callable[[T, E], E]) -> Callable[[Iterator[X], ENV], ENV]:


def oho(t: Union[HasParameter[ENV, T], HasHyperParameter[ENV, E]], ohoT: Callable[[X, T, E], E]) -> Callable[[X, ENV], ENV]:
    def oho_(x: X, env: ENV) -> ENV:
        p = t.getParameter(env)
        hp = t.getHyperParameter(env)
        hp_ = ohoT(x, p, hp)
        return t.putHyperParameter(hp_, env)
    return oho_


#
"""
TODO
1) offline rnn
2) offline rnn + oho
3) online rnn + oho 
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



def rnnT(activation: Callable) -> Callable[[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor], torch.Tensor]:
    def rnnT_(x: torch.Tensor, param: tuple[torch.Tensor, torch.Tensor, torch.Tensor, float], h: torch.Tensor) -> torch.Tensor:
        W_in, W_rec, b_rec, alpha = param
        return (1 - alpha) * h + alpha * activation(f.linear(x, W_in, None) + f.linear(h, W_rec, b_rec))
    return rnnT_

test = rnnT(f.relu)
test2 = rnnStep(OhoState(), test)