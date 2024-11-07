import itertools
from typing import TypeVar, Callable, Generic, Generator, Iterator, Any
from functools import reduce
from toolz.curried import curry, map, concat, compose
import torch
import numpy as np
from torch.nn import functional as f

T = TypeVar('T') 
X = TypeVar('X')
Y = TypeVar('Y')
A = TypeVar('A') 
B = TypeVar('B')
C = TypeVar('C')
D = TypeVar('D')

"""
-- 1) ffwd step
ffwd :: a -> (p, hp) -> (p, hp)
ffwd = undefined

-- 2) normal ffwd training
ffwdTrain = trans' (Proxy :: Proxy []) ffwd
"""

def feedfwdTrain(lossFn: Callable[[A, C], Any], x: C, pair: tuple[A, B]) -> tuple[A, B]:
    param, hparam = pair 
    lossFn(param, x)

"""t3'' :: x -> hp -> p -> hp
t3'' = undefined"""

def oho()