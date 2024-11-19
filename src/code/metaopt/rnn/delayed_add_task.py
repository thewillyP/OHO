
#%% 
import numpy as np
from typing import Callable
from toolz import curry, compose
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import TypeVar, Callable, Generic, Generator, Iterator
from dataclasses import dataclass
import hashlib


T = TypeVar('T')
X = TypeVar('X')

def sampleTimeSeries(mapper, ts: Iterator[T], f: Callable[[T], X]) -> Iterator[X]:
    return mapper(f)(ts)

def sampleTimeSeriesTorch(ts: Iterator[T], f: Callable[[T], X]) -> torch.Tensor:
    return sampleTimeSeries(torch.vmap, torch.tensor(list(ts), dtype=torch.float32), f)

@curry
def createUnitSignal(startTime: float, duration: float) -> Callable[[float], float]:
    def unit(t):
        return ((0 <= t - startTime) & (t - startTime <= duration)) 
    return unit

@curry 
def createSparseSignal(amplitude: float, t0: float, duration: float):
    return compose(lambda x: amplitude*x, createUnitSignal(t0, duration))

@curry
def createDelayedAdder(  t1: float
                        , t2: float
                        , x1: Callable[[float], float]
                        , x2: Callable[[float], float]) -> Callable[[float], float]:
    return lambda t: x1(t - t1) + x2(t - t2)


def sinWave(amplitude: float, frequency: float, phase_shift: float, bias: float):
    return lambda t: amplitude * torch.sin(frequency * t + phase_shift) + bias

def sparseSignal( startOffset: float
                , a: float
                , dur: float
                , outT: float):
    return createSparseSignal(a, outT - startOffset, dur)

@dataclass(frozen=True)
class WaveInitIO:
    initAmplitudeIO: Callable[[torch.Generator], float]
    initFrequencyIO: Callable[[torch.Generator], float]
    initPhaseShiftIO: Callable[[torch.Generator], float]
    initBiasIO: Callable[[torch.Generator], float]

@dataclass(frozen=True)
class SparseInitIO:
    initAmplitudeIO: Callable[[torch.Generator], float]
    initT0IO: Callable[[torch.Generator], float]
    initDurationIO: Callable[[torch.Generator], float]

@dataclass(frozen=True)
class RandomInitIO:
    randomFnIO: Callable[[torch.Generator], float]

@curry 
def waveIO(waveInitIO: WaveInitIO, globalSeed: int, state: int):
    generator = hashSeed((globalSeed, state))
    amplitude = waveInitIO.initAmplitudeIO(generator)
    frequency = waveInitIO.initFrequencyIO(generator)
    phase_shift = waveInitIO.initPhaseShiftIO(generator)
    bias = waveInitIO.initBiasIO(generator)
    return sinWave(amplitude, frequency, phase_shift, bias)

@curry 
def sparseIO(sparseInitIO: SparseInitIO, globalSeed: int, state: int):
    generator = hashSeed((globalSeed, state))
    amplitude = sparseInitIO.initAmplitudeIO(generator)
    t0 = sparseInitIO.initT0IO(generator)
    duration = sparseInitIO.initDurationIO(generator)
    return sparseSignal(amplitude, t0, duration)

@curry
def randomFnIO(randomInitIO: RandomInitIO, globalSeed: int, state: int):
    def rf(x):
        seed = (globalSeed, state, x)
        generator = hashSeed(seed)
        return randomInitIO.randomFnIO(generator)
    return rf


def randomFnIO(randomInitIO: RandomInitIO, generator):
    personale = {}
    def rf(x):
        if x not in personale:
            personale[x] = randomInitIO.randomFnIO(generator)
        return personale[x]
    return rf
    


# def randomFunctionGeneratorIO(randFn: Callable[[int, int], Callable], globalSeed: int):
#     state = 1
#     while True:
#         yield randFn(globalSeed, state)
#         state += 1


def delayedAddGeneratorIO(t1: float, t2: float, randFn: Callable[[int, int], Callable], globalSeed: int, state1: int, state2: int):
    x1 = randFn(globalSeed, state1)
    x2 = randFn(globalSeed, state2)
    y = createDelayedAdder(t1, t2, x1, x2)
    return x1, x2, y


def createExamples(t1, t2, numExamples, samples, randFn, globalSeed):
    def generateExample(state):
        s1, s2 = state
        x1, x2, y = delayedAddGeneratorIO(t1, t2, randFn, globalSeed, s1, s2)
        return torch.vmap(x1)(samples), torch.vmap(x2)(samples), torch.vmap(y)(samples)
    
    examples = torch.arange(0, numExamples * 2).reshape(numExamples, 2)
    XS1, XS2, YS = torch.vmap(generateExample)(examples)
    combined_XS = torch.stack((XS1, XS2), dim=-1)
    return combined_XS, YS


random_init = RandomInitIO(lambda gen: torch.randn(1, generator=gen).item())
random_fn = randomFnIO(random_init)
XS, YS = createExamples(3, 2, 3, torch.linspace(0, 15, 30), random_fn, 0)
print(XS.shape, YS.shape)


# @curry
# def createExamples(ts, x1, x2, y):
#     # x1s = tensormap(x1, ts)
#     # x2s = tensormap(x2, ts) 
#     # ys = tensormap(y, ts)
#     x1s = x1(ts)
#     x2s = x2(ts)
#     ys = y(ts)
#     return torch.stack([x1s, x2s], dim=1), ys

# def createExamplesIO(numExamples, ts, randomFnsIO):
#     X_batch = []
#     Y_batch = []
    
#     for _ in range(numExamples):
#         x1, x2, y = randomFnsIO()
#         X, Y = createExamples(ts, x1, x2, y)
#         X_batch.append(X)
#         Y_batch.append(Y)
    
#     X_batch = torch.stack(X_batch, dim=0)
#     Y_batch = torch.stack(Y_batch, dim=0).unsqueeze(-1)
    
#     return X_batch, Y_batch


# def createExamplesIO2(numExamples, ts, createExamplesIO):
#     X_batch = []
#     Y_batch = []
    
#     for _ in range(numExamples):
#         fn = createExamplesIO()
#         X, Y = fn(ts)
#         X_batch.append(X)
#         Y_batch.append(Y)
    
#     X_batch = torch.stack(X_batch, dim=0)
#     Y_batch = torch.stack(Y_batch, dim=0).unsqueeze(-1)
    
#     return X_batch, Y_batch




# def createSparseAddMemoryTask(t1: float
#                             , t2: float
#                             , a: float
#                             , b: float
#                             , t1_dur: float
#                             , t2_dur: float
#                             , outT: float):
#     x1 = createSparseSignal(a, outT - t1, t1_dur)
#     x2 = createSparseSignal(b, outT - t2, t2_dur)
#     return x1, x2, createDelayedAdder(t1, t2, x1, x2)






#%%


# if __name__ == "__main__":

#     np.random.seed(0)  #! Global state change
#     t1: float = 6
#     t2: float = 4
#     ts = torch.linspace(0, 15, 1000)
#     batch = 2


    

#     genRndomSineExampleIO = lambda: randomSineExampleIO(t1, t2)



#     xs, ys = createExamplesIO(3, ts, genRndomSineExampleIO)
#     dataset = TensorDataset(xs, ys)
#     dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)

#     x1, x2, y = genRndomSineExampleIO()
#     plt.plot(ts, listmap(x1, ts), ts, listmap(x2, ts), ts, listmap(y, ts))








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
