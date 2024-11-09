from typing import TypeVar, Callable, Generic, Generator, Iterator
from functools import reduce
from toolz.curried import curry, map, concat, compose
import itertools
import numpy as np
import torch 
from torch.nn import functional as f


T = TypeVar('T')
E = TypeVar('E')
A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')
D = TypeVar('D')
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


cycle_efficient = compose(itertools.chain.from_iterable, itertools.repeat)


def composeSnd(f: Callable[[A, B], C], g: Callable[[C], D]) -> Callable[[A, B], D]:
    def composeSnd_(a: A, b: B) -> D:
        return g(f(a, b))
    return composeSnd_


def liftA2(f: Callable[[A, B], C], g: Callable[[D], A], h: Callable[[D], B]) -> Callable[[D], C]:
    def liftA2_(d: D) -> C:
        return f(g(d), h(d))
    return liftA2_

# def liftA1(f: Callable[[B], C], g: Callable[[A], B]) -> Callable[[A], C]:
#     return flip(compose2)

def flip(f: Callable[[A, B], C]) -> Callable[[B, A], C]:
    def flip_(b: B, a: A) -> C:
        return f(a, b)
    return flip_

def apply(f: Callable[[A], B], x: A) -> B:
    return f(x)

def const(x: A) -> Callable[[B], A]:
    return lambda _: x

def compose2(f: Callable[[A], B], g: Callable[[B], C]) -> Callable[[A], C]:
    def compose2_(a: A) -> C:
        return g(f(a))
    return compose2_

fmap = flip(compose2)



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


linear_ = curry(lambda w, b, h: f.linear(h, w, b))


@curry
def rnnTransition(W_in, W_rec, b_rec, activation, alpha, h, x):
    return (1 - alpha) * h + alpha * activation(f.linear(x, W_in, None) + f.linear(h, W_rec, b_rec))