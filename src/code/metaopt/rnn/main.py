
import torch 
from torch.nn import functional as f
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from typing import TypeVar, Callable, Generic, Generator, Iterator
from toolz.curried import curry, compose, identity, take_nth, accumulate, apply, map, concat, take, drop, mapcat, last
from functools import reduce
import torchvision
import torchvision.transforms as transforms
from itertools import tee
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from func import *

dataset = datasets.MNIST('data/mnist', train=True, download=True,
                                        transform=transforms.Compose(
                                                [transforms.ToTensor()]))
train_set, valid_set = torch.utils.data.random_split(dataset,[60000 - 10000, 10000])
data_loader_tr = DataLoader(train_set, batch_size=100, shuffle=True)
data_loader_vl = DataLoader(valid_set, batch_size=100, shuffle=True)
data_loader_te = DataLoader(datasets.MNIST('data/mnist', train=False, download=True,
                                                    transform=transforms.Compose(
                                                            [transforms.ToTensor()])),
                                            batch_size=100, shuffle=True)
data_loader_vl = cycle_efficient(data_loader_vl)



num_classes = 10
num_epochs = 10
batch_size = 100

input_size = 28
sequence_length = 28
hidden_size = 128
num_layers = 2

alpha_ = 1
activation_ = f.relu
learning_rate = 0.001



W_rec_, W_in_, b_rec_, W_out_, b_out_ = initializeParametersIO(input_size, hidden_size, num_classes)
readout = linear_(W_out_, b_out_)
optimizer = torch.optim.Adam([W_rec_, W_in_, b_rec_, W_out_, b_out_], lr=learning_rate)  


p0 = rnnTransition(W_in_, W_rec_, b_rec_, activation_, alpha_)
h0 = torch.zeros(batch_size, hidden_size, dtype=torch.float32)
stateM0 = (-1, h0, (p0, readout))
putState = composeST( backPropAt(sequence_length, updateParameterState(optimizer, f.cross_entropy)) # 😱😱😱😱😱
        ,  composeST( resetHiddenStateAt(sequence_length, h0)
                    , incrementCounter))
getHiddenStates = nonAutonomousStateful(  updateHiddenState
                                        , noParamUpdate  
                                        , putState) # 😱😱😱😱😱



# hiddenState
"""
OHO:
parameterUpdate(grad, gradient, input, hiddenState)
"""

# @curry
# def backPropAt(n0, parameterUpdateFn, s, fn1, fn2):
#     return (s, fn1, parameterUpdateFn) if s > 0 and (s+1) % n0 == 0 else (s, fn1, fn2)  # s+1 bc backprop right before hidden state reset


# noParamUpdate = lambda _, fp, __: fp

# @curry
# def updateHiddenState(h, parameters, observation):
#     x, _ = observation
#     forwProp, _ = parameters
#     return forwProp(h, x)


# step = 0

# # TODO: Figure out how to make this purely functional alter. 
# @curry
# def updateParameterState(optimizer, lossFn, h, parameters, observation):
#     global step # 😱😱😱😱😱

#     _, label = observation


#     if label is not None:  # None is a substitute for the Maybe monad for now
#         _, readout = parameters
#         loss = lossFn(readout(h), label)
#         optimizer.zero_grad()  ### 😱😱😱😱
#         loss.backward()  # 😱😱😱😱
#         optimizer.step() # 😱😱😱😱😱

#         # 😱😱😱
#         if (step+1) % 5000 == 0:
#             print (f'Step [{step+1}], Loss: {loss.item():.10f}')
#         step += 1
#     return parameters  # autograd state implictly updates these guys. 


class RTRL(Learning_Algorithm):
    """Implements the Real-Time Recurrent Learning (RTRL) algorithm from
    Williams and Zipser 1989.

    RTRL maintains a long-term "influence matrix" dadw that represents the
    derivative of the hidden state with respect to a flattened vector of
    recurrent update parameters. We concatenate [W_rec, W_in, b_rec] along
    the column axis and order the flattened vector of parameters by stacking
    the columns end-to-end. In other words, w_k = W_{ij} when i = k%n_h and
    j = k//n_h. The influence matrix updates according to the equation

    M' = JM + M_immediate                            (1)

    where J is the network Jacobian and M_immediate is the immediate influence
    of a parameter w on the hidden state a. (See paper for more detailed
    notation.) M_immediate is notated as papw in the code for "partial a partial
    w." For a vanilla network, this can be simply (if inefficiently) computed as
    the Kronecker product of a_hat = [a_prev, x, 1] (a concatenation of the prev
    hidden state, the input, and a constant 1 (for bias)) with the activation
    derivatives organized in a diagonal matrix. The implementation of Eq. (1)
    is in the update_learning_vars method.

    Finally, the algorithm returns recurrent gradients by projecting the
    feedback vector q onto the influence matrix M:

    dL/dw = dL/da da/dw = qM                         (2)

    Eq. (2) is implemented in the get_rec_grads method."""

    def __init__(self, rnn, M_decay=1, **kwargs):
        """Inits an RTRL instance by setting the initial dadw matrix to zero."""

        self.name = 'RTRL' #Algorithm name
        allowed_kwargs_ = set() #No special kwargs for RTRL
        super().__init__(rnn, allowed_kwargs_, **kwargs)

        #Initialize influence matrix
        self.dadw = np.zeros((self.n_h, self.rnn.n_h_params))
        self.M_decay = M_decay

    def update_learning_vars(self):
        """Updates the influence matrix via Eq. (1)."""

        #Get relevant values and derivatives from network.
        self.a_hat = np.concatenate([self.rnn.a_prev,
                                     self.rnn.x,
                                     np.array([1])])
        D = self.rnn.alpha * np.diag(self.rnn.activation.f_prime(self.rnn.h))
        self.papw = np.kron(self.a_hat, D) #Calculate M_immediate
        self.rnn.get_a_jacobian() #Get updated network Jacobian

        #Update influence matrix via Eq. (1).
        self.dadw = self.M_decay * self.rnn.a_J.dot(self.dadw) + self.papw

    def get_rec_grads(self):
        """Calculates recurrent grads using Eq. (2), reshapes into original
        matrix form."""

        return self.q.dot(self.dadw).reshape((self.n_h, self.m), order='F')

    def reset_learning(self):
        """Resets learning algorithm by setting influence matrix to 0."""

        self.dadw *= 0



image2Rows = compose(traverseTuple, mapTuple1(lambda image: image.reshape(-1, sequence_length, input_size).permute(1, 0, 2)))  # [28, N, 28] -> [N, 28] (batch, input vector)

getOutputs = lambda initState: compose(map(hideStateful)
                                    , drop(1)  # non-autnonmous scan returns h0 w/o +input, whose readout we don't care
                                    , take_nth(sequence_length)
                                    , getHiddenStates(initState)  
                                    , mapcat(image2Rows)) 

outputs = getOutputs(stateM0)
doEpochs = mapcat(outputs)
epochs = epochsIO(num_epochs, train_loader)

_, (pN, readoutN) = last(doEpochs(epochs))


def predict(output, target):
    _, predicted = torch.max(output.data, 1)
    n_samples = target.size(0)
    n_correct = (predicted == target).sum().item()
    return (n_samples, n_correct)


accuracy = compose(   lambda pair: 100.0 * pair[1] / pair[0]
                    , totalStatistic(predict, lambda res, pair: (res[0] + pair[0], res[1] + pair[1])))

with torch.no_grad():
    xs_test, ys_test = tee(test_loader, 2)
    xtream_test, targets_test = map(compose(lambda x: (x, None), fst), xs_test), map(snd, ys_test)

    def getReadout(pair):
        h, (_, rd) = pair 
        return rd(h)
    testOuputs = compose( map(getReadout)
                        , getOutputs((-1, h0, (pN, readoutN))))
    print(accuracy(testOuputs, xtream_test, targets_test))





