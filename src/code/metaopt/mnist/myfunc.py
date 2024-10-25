from typing import TypeVar, Callable, Generic, Generator, Iterator
from functools import reduce
from toolz.curried import curry, map, concat, compose
import torch
import numpy as np
from torch.nn import functional as f


T = TypeVar('T') 
X = TypeVar('X')
Y = TypeVar('Y')

@curry
def scan(f: Callable[[T, X], T], state: T, it: Iterator[X]) -> Generator[T, None, None]:
    yield state
    for x in it:
        state = f(state, x)
        yield state


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