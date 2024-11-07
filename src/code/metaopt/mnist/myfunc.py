import itertools
from typing import TypeVar, Callable, Generic, Generator, Iterator
from functools import reduce
from toolz.curried import curry, map, concat, compose
import torch
import numpy as np
from torch.nn import functional as f
from operator import add


T = TypeVar('T') 
X = TypeVar('X')
Y = TypeVar('Y')
A = TypeVar('A') 
B = TypeVar('B')
C = TypeVar('C')
D = TypeVar('D')


@curry
def scan(f: Callable[[T, X], T], state: T, it: Iterator[X]) -> Generator[T, None, None]:
    yield state
    for x in it:
        state = f(state, x)
        yield state

@curry 
def flip(f: Callable[[A, B], C]) -> Callable[[B, A], C]:
    def _flip(b, a):
        return f(a, b)
    return _flip

@curry 
def const(x: X, _: Y) -> X:
    return x

@curry
def uncurry(f: Callable[[T, X], Y]) -> Callable[[tuple[T, X]], Y]:
    def _uncurry(pair):
        x, y = pair 
        return f(x, y)
    return _uncurry

@curry 
def swap(f: Callable[[X, Y], T]) -> Callable[[Y, X], T]:
    def swap_(y, x):
        return f(x, y)
    return swap_

@curry
def map2(f1: Callable, f2: Callable):
    return map(uncurry(lambda x, y: (f1(x), f2(y))))

def fst(pair: tuple[X, Y]) -> X:
    x, _ = pair 
    return x

def snd(pair: tuple[X, Y]) -> Y:
    _, y = pair 
    return y

# def flatFst(stream: Iterator[tuple[Iterator[X], Y]]) -> Iterator[tuple[X, Y]]:
#     i1 = concat(map(fst, stream))
#     i2 = map(snd, stream)
#     return map(lambda x, y: (x, y), i1, i2)


reduce_ = curry(lambda fn, x, xs: reduce(fn, xs, x))

foldr = curry(lambda fn, xs, x: reduce(fn, xs, x))


# reverse of sequenceA? which doesn't exist so custom logic
@curry
def traverseTuple(pair: tuple[Iterator[X], Y]) -> Iterator[tuple[X, Y]]:
    xs, y = pair 
    return ((x, y) for x in xs)


@curry
def mapTuple1(f, pair):
    a, b = pair 
    return (f(a), b)


listmap = compose(list, map)

tensormap = compose(torch.tensor, listmap)


def jacobian(_os, _is):
    I_N = torch.eye(len(_os))
    def get_vjp(v):
        return torch.autograd.grad(_os, _is, v)
    return torch.vmap(get_vjp)(I_N)


@curry
def initializeParametersIO(n_in: int, n_h: int, n_out: int
                        ) -> tuple[torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter]:
    W_in = np.random.normal(0, np.sqrt(1/(n_in)), (n_h, n_in))
    W_rec = np.linalg.qr(np.random.normal(0, 1, (n_h, n_h)))[0]
    W_out = np.random.normal(0, np.sqrt(1/(n_h)), (n_out, n_h))
    b_rec = np.random.normal(0, np.sqrt(1/(n_h)), (n_h,))
    b_out = np.random.normal(0, np.sqrt(1/(n_out)), (n_out,))

    # _W_rec = torch.nn.Parameter(torch.tensor(W_rec, requires_grad=True, dtype=torch.float32))
    # _W_in = torch.nn.Parameter(torch.tensor(W_in, requires_grad=True, dtype=torch.float32))
    # _b_rec = torch.nn.Parameter(torch.tensor(b_rec, requires_grad=True, dtype=torch.float32))
    # _W_out = torch.nn.Parameter(torch.tensor(W_out, requires_grad=True, dtype=torch.float32))
    # _b_out = torch.nn.Parameter(torch.tensor(b_out, requires_grad=True, dtype=torch.float32))
    _W_rec = torch.nn.Parameter(torch.from_numpy(W_rec).float().requires_grad_())
    _W_in = torch.nn.Parameter(torch.from_numpy(W_in).float().requires_grad_())
    _b_rec = torch.nn.Parameter(torch.from_numpy(b_rec).float().requires_grad_())
    _W_out = torch.nn.Parameter(torch.from_numpy(W_out).float().requires_grad_())
    _b_out = torch.nn.Parameter(torch.from_numpy(b_out).float().requires_grad_())

    return _W_rec, _W_in, _b_rec, _W_out, _b_out

@curry
def rnnTransition(W_in, W_rec, b_rec, activation, alpha, h, x):
    return (1 - alpha) * h + alpha * activation(f.linear(x, W_in, None) + f.linear(h, W_rec, b_rec))



linear_ = curry(lambda w, b, h: f.linear(h, w, b))

cycle_efficient = compose(itertools.chain.from_iterable, itertools.repeat)

def getEpochs(numEpochs, *dataLoaders):
    return itertools.chain.from_iterable((zip(*dataLoaders) for _ in range(numEpochs)))





"""unfusei :: (a -> b -> b) -> (c -> b -> b) -> (a, c) -> b -> b
unfusei f g (x, y) = f x . g y -- use to combine when inputs are different"""

@curry 
def multiplex(f: Callable[[A, B], B], g: Callable[[C, B], B], pair: tuple[A, C]) -> Callable[[B], B]:
    x, y = pair 
    return compose(g(y), f(x))

"""trans :: (Foldable t) => (a -> b -> b) -> t a -> b -> b
trans fn = foldr ((.) . fn) id"""

# @curry 
# def trans(fn: Callable[[A, B], B], xs: Iterator[A], b0: B) -> B:
#     return reduce_(compose(compose, fn), lambda x: x, b0, xs)


# print(foldr(add, range(1, 4)))


# @curry
# def dstep(f: Callable[[A, B], C], g: Callable[[C, B], D], pair: tuple[A, B]) -> tuple[C, D] :
#     """
#     dstep :: (a1 -> b1 -> a2) -> (a2 -> b1 -> b2) -> (a1, b1) -> (a2, b2)
#     dstep f g (a, b) = let a' = f a b
#                         b' = g a' b
#                     in (a', b')
#     """
#     a, b = pair 
#     a_ = f(a, b)
#     b_ = g(a_, b)
#     return (a_, b_)


# """ yy :: (a -> a1 -> b1 -> a2) -> (a2 -> b1 -> b2) -> a -> (a1, b1) -> (a2, b2)
# yy f g = f &&& const g >>> uncurry test
# fmapCombine :: (c1 -> c' -> c2) -> (a -> c1) -> c' -> a -> c2
#  """
# """ &&& = \f g x -> (f x, g x) """

# @curry
# def AAA(f: Callable[[A], B], g: Callable[[A], C], x: A) -> tuple[B, C]:
#     return (f(x), g(x))

# @curry 
# def fmapCombine(combiner: Callable[[A, B], C], combinable: Callable[[D], A], combinee: B) -> Callable[[D], C]:
#     return compose(uncurry(combiner), AAA(combinable, const(combinee)))

# # the main reason why I want this scaffolding is because I want to code oho once and I can always attach it to any function :: x -> hp -> p -> p without writing any extra code. 
