from typing import TypeVar, Callable, Generic, Generator, Iterator
from functools import reduce
from toolz.curried import curry, map, concat, compose
import torch

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