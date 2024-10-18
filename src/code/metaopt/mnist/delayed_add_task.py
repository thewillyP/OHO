
#%% 
import numpy as np
from typing import Callable
from toolz import curry, compose
import matplotlib.pyplot as plt
from myfunc import *
import torch
from torch.utils.data import TensorDataset, DataLoader

@curry
def createUnitSignal(startTime: float, duration: float) -> Callable[[float], float]:
    def test(t):
        return 1.0 * float(0 <= t - startTime <= duration) 
    return test
    # return lambda t: 1.0 * (0 <= t - startTime <= duration) 

@curry 
def createSparseSignal(amplitude: float, t0: float, duration: float):
    return compose(lambda x: amplitude*x, createUnitSignal(t0, duration))


@curry
def createDelayedAdder(  t1: float
                        , t2: float
                        , x1: Callable[[float], float]
                        , x2: Callable[[float], float]) -> Callable[[float], float]:
    return lambda t: x1(t - t1) + x2(t - t2)


def createAddMemoryTask(  t1: float
                        , t2: float
                        , a: float
                        , b: float
                        , t1_dur: float
                        , t2_dur: float
                        , outT: float):
    x1 = createSparseSignal(a, outT - t1, t1_dur)
    x2 = createSparseSignal(b, outT - t2, t2_dur)
    return x1, x2, createDelayedAdder(t1, t2, x1, x2)


@curry
def createExamples(ts, x1, x2, y):
    # x1s = tensormap(x1, ts)
    # x2s = tensormap(x2, ts) 
    # ys = tensormap(y, ts)
    x1s = x1(ts)
    x2s = x2(ts)
    ys = y(ts)
    return torch.stack([x1s, x2s], dim=1), ys

def createExamplesIO(numExamples, ts, randomFnsIO):
    X_batch = []
    Y_batch = []
    
    for _ in range(numExamples):
        x1, x2, y = randomFnsIO()
        X, Y = createExamples(ts, x1, x2, y)
        X_batch.append(X)
        Y_batch.append(Y)
    
    X_batch = torch.stack(X_batch, dim=0)
    Y_batch = torch.stack(Y_batch, dim=0).unsqueeze(-1)
    
    return X_batch, Y_batch


#%%


if __name__ == "__main__":

    np.random.seed(0)  #! Global state change
    t1: float = 6
    t2: float = 4
    ts = torch.linspace(0, 15, 1000)
    batch = 2


    def randomSineWaveIO():
        amplitude = np.random.uniform(-1, 1)  # Random amplitude between 0.5 and 2
        frequency = np.random.uniform(0, 100)   # Random frequency between 1 and 10 Hz
        phase_shift = np.random.uniform(0, 2 * np.pi)
        bias = np.random.uniform(-1, 1)
        sine_wave = lambda t: amplitude * np.sin(frequency * t + phase_shift) + bias
        return sine_wave

    def randomSineExampleIO(t1: float, t2: float):
        x1 = randomSineWaveIO()
        x2 = randomSineWaveIO()
        y = createDelayedAdder(t1, t2, x1, x2)
        return x1, x2, y

    genRndomSineExampleIO = lambda: randomSineExampleIO(t1, t2)



    xs, ys = createExamplesIO(3, ts, genRndomSineExampleIO)
    dataset = TensorDataset(xs, ys)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)

    x1, x2, y = genRndomSineExampleIO()
    plt.plot(ts, listmap(x1, ts), ts, listmap(x2, ts), ts, listmap(y, ts))








# a: float = 3
# b: float = -1
# t1_dur: float = 2
# t2_dur: float = 1
# outT: float = 10
# st, et = min(outT - t1, outT - t2), outT+1  # +1 to include outT in range()

# x1, x2, y = createAddMemoryTask(t1, t2, a, b, t1_dur, t2_dur, outT)


# x1, x2, y = createAddMemoryTask(t1, t2, a, b, t1_dur, t2_dur, outT)
# ts = np.linspace(-10, 15, 1000)
# x1s = listmap(x1, ts)
# x2s = listmap(x2, ts) 
# ys = listmap(y, ts)
# plt.plot(ts, x1s, ts, x2s, ts, ys)
# plt.show()


# """
#     y(t) = x(t - t_1) + x(t - t_2)           (1)
# """
# # :: time (<No prev state bc stateless>, Action) -> (x1, x2, y) (State)
# @curry
# def createAddMemoryTask(  t1: float
#                         , t2: float
#                         , a: float
#                         , b: float
#                         , t1_dur: float
#                         , t2_dur: float
#                         , outT: float) -> Callable[[float], tuple[float, float, float]]:
#     x1 = compose(lambda x: a*x, createUnitSignal(outT - t1, t1_dur))
#     x2 = compose(lambda x: b*x, createUnitSignal(outT - t2, t2_dur))
#     y = lambda t: x1(t - t1) + x2(t - t2)
#     return lambda t: (x1(t), x2(t), y(t))






# %%
