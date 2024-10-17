

import numpy as np
from typing import Callable
from toolz import curry, compose

@curry
def createUnitSignal(startTime: float, duration: float) -> Callable[[float], float]:
    def test(t):
        return 1.0 * float(0 <= t - startTime <= duration) 
    return test
    # return lambda t: 1.0 * (0 <= t - startTime <= duration) 

"""
    y(t) = x(t - t_1) + x(t - t_2)           (1)
"""
# :: time (<No prev state bc stateless>, Action) -> (x1, x2, y) (State)
@curry
def createAddMemoryTask(  t1: float
                        , t2: float
                        , a: float
                        , b: float
                        , t1_dur: float
                        , t2_dur: float
                        , outT: float) -> Callable[[float], tuple[float, float, float]]:
    x1 = compose(lambda x: a*x, createUnitSignal(outT - t1, t1_dur))
    x2 = compose(lambda x: b*x, createUnitSignal(outT - t2, t2_dur))
    y = lambda t: x1(t - t1) + x2(t - t2)
    return lambda t: (x1(t), x2(t), y(t))






