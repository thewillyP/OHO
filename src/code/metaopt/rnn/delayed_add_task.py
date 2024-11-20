
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

# @curry
# def createDelayedAdder(  t1: float
#                         , t2: float
#                         , x1: Callable[[float], float]
#                         , x2: Callable[[float], float]) -> Callable[[float], float]:
#     return lambda t: x1(t - t1) + x2(t - t2)


def sinWave(amplitude: float, frequency: float, phase_shift: float, bias: float):
    return lambda t: amplitude * torch.sin(frequency * t + phase_shift) + bias

def sparseSignal( startOffset: float
                , a: float
                , dur: float
                , outT: float):
    return createSparseSignal(a, outT - startOffset, dur)

# @dataclass(frozen=True)
# class WaveInitIO:
#     initAmplitudeIO: Callable[[torch.Generator], float]
#     initFrequencyIO: Callable[[torch.Generator], float]
#     initPhaseShiftIO: Callable[[torch.Generator], float]
#     initBiasIO: Callable[[torch.Generator], float]

# @dataclass(frozen=True)
# class SparseInitIO:
#     initAmplitudeIO: Callable[[torch.Generator], float]
#     initT0IO: Callable[[torch.Generator], float]
#     initDurationIO: Callable[[torch.Generator], float]

# @dataclass(frozen=True)
# class RandomInitIO:
#     randomFnIO: Callable[[torch.Generator], float]

# @curry 
# def waveIO(waveInitIO: WaveInitIO, generator: torch.Generator):
#     amplitude = waveInitIO.initAmplitudeIO(generator)
#     frequency = waveInitIO.initFrequencyIO(generator)
#     phase_shift = waveInitIO.initPhaseShiftIO(generator)
#     bias = waveInitIO.initBiasIO(generator)
#     return sinWave(amplitude, frequency, phase_shift, bias)

# @curry 
# def sparseIO(sparseInitIO: SparseInitIO, generator: torch.Generator):
#     amplitude = sparseInitIO.initAmplitudeIO(generator)
#     t0 = sparseInitIO.initT0IO(generator)
#     duration = sparseInitIO.initDurationIO(generator)
#     return sparseSignal(amplitude, t0, duration)

# @curry
# def randomFnIO(randomInitIO: RandomInitIO, generator: torch.Generator):
#     return lambda _: randomInitIO.randomFnIO(generator)

@dataclass(frozen=True)
class WaveInitIO:
    initAmplitudeIO: Callable[[], float]
    initFrequencyIO: Callable[[], float]
    initPhaseShiftIO: Callable[[], float]
    initBiasIO: Callable[[], float]
@dataclass(frozen=True)
class SparseInitIO:
    initAmplitudeIO: Callable[[], float]
    initT0IO: Callable[[float], float]
    initDurationIO: Callable[[], float]
    outTIO : Callable[[], float]


def waveIO(waveInitIO: WaveInitIO):
    amplitude = waveInitIO.initAmplitudeIO()
    frequency = waveInitIO.initFrequencyIO()
    phase_shift = waveInitIO.initPhaseShiftIO()
    bias = waveInitIO.initBiasIO()
    return lambda t, _: sinWave(amplitude, frequency, phase_shift, bias)(t)


def sparseIO(sparseInitIO: SparseInitIO):
    outT = sparseInitIO.outTIO()
    def sparseFn(ts, t0):
        t0 = sparseInitIO.initT0IO(t0)
        amplitude = sparseInitIO.initAmplitudeIO()
        duration = sparseInitIO.initDurationIO()
        return sparseSignal(t0, amplitude, duration, outT)(ts)
    return sparseFn, sparseFn

def randomIO(ts, _):
    return torch.randn(len(ts))

def createExample(t1, t2, ts, getRandFn):
    randFn1, randFn2 = getRandFn()
    x1 = randFn1(ts, t1)
    x2 = randFn2(ts, t2)
    y = torch.roll(x1, shifts=t1, dims=0) + torch.roll(x2, shifts=t2, dims=0)  # giving up on mathematically elegant y(t) = x(t - t1) + x(t - t2) bc computer can't support arbitrary precision. From now use integers only
    mask = ts >= max(t1, t2) 
    y = y * mask 
    return x1, x2, y


def createExamples(n, randomMonad):
    XS1, XS2, YS = torch.vmap(lambda _: randomMonad(), randomness='different')(torch.arange(n))
    combined_XS = torch.stack((XS1, XS2), dim=-1)
    return combined_XS, YS.unsqueeze(-1)


torch.manual_seed(0)

s1 = SparseInitIO(lambda: torch.rand(1) - 0.5, lambda x: x, lambda: 1, lambda: 8)
s2 = SparseInitIO(lambda: torch.rand(1) - 0.5, lambda x: x, lambda: 1, lambda: torch.randint(5, 10, (1,)))
w1 = WaveInitIO(lambda: torch.rand(1), lambda: torch.rand(1)*100, lambda: torch.rand(1)*2*torch.pi, lambda: torch.rand(1)*2 - 1)

randFn1 = randomIO
randFn2 = randomIO
getRandFn = lambda: (randFn1, randFn2)

getRandFn = lambda: sparseIO(s1)

getRandFn = lambda: (waveIO(w1), waveIO(w1))

gen = lambda: createExample(5, 1, torch.arange(20), getRandFn)

XS, YS = createExamples(50, gen)







# def delayedAddGeneratorIO(t1: float, t2: float, samples: torch.Tensor, randFn1, randFn2):
#     x1 = randFn1(samples, t1)
#     x2 = randFn2(samples, t2)
#     y = torch.roll(x1, shifts=t1, dims=0) + torch.roll(x2, shifts=t2, dims=0)  # giving up on mathematically elegant y(t) = x(t - t1) + x(t - t2) bc computer can't support arbitrary precision. From now use integers only
#     mask = samples >= max(t1, t2) 
#     y = y * mask 
#     return x1, x2, y


# def createExamples(t1, t2, numExamples, samples, randFn):
#     XS1, XS2, YS = torch.vmap(lambda _: delayedAddGeneratorIO(t1, t2, samples, randFn), randomness='different')(torch.arange(numExamples))
#     combined_XS = torch.stack((XS1, XS2), dim=-1)
#     return combined_XS, YS.unsqueeze(-1)
    



# print(XS.shape, YS.shape)
# print(XS[0])
# print(YS[0])

# for i in range(XS.size(0)):  # Loop over each example
#     plt.figure(figsize=(10, 6))
    
#     # Plot features from X (features are in the 3rd dimension, here we plot both)
#     plt.plot(range(XS.size(1)), XS[i, :, 0].numpy(), label=f'Feature 1 of Example {i+1}', color='b')
#     plt.plot(range(XS.size(1)), XS[i, :, 1].numpy(), label=f'Feature 2 of Example {i+1}', color='g')
    
#     # Plot the output Y for the same example
#     plt.plot(range(YS.size(1)), YS[i, :, 0].numpy(), label=f'Output of Example {i+1}', color='r')
    
#     plt.title(f'Time Series for Example {i+1}')
#     plt.xlabel('Time')
#     plt.ylabel('Value')
#     plt.legend()
#     plt.show()

from matplotlib.ticker import MaxNLocator

for i in range(YS.size(0)):  # Loop over each example
    plt.plot(range(YS.size(1)), YS[i, :, 0].numpy(), 'o-', label=f'Example {i+1}')


plt.title('Output Time Series for All Examples')
plt.xlabel('Time')
plt.ylabel('Output Value')
# plt.legend()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()




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
