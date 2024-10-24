#%%
import torch
from torch.nn import functional as f
from toolz.curried import curry, compose, identity, take_nth, accumulate, apply, map, concat, take, drop
import numpy as np
from myfunc import scan
from torch.autograd import gradcheck

torch.manual_seed(0)
np.random.seed(0)

@curry
def initializeParametersIO(n_in: int, n_h: int, n_out: int
                        ) -> tuple[torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter]:
    W_in = np.random.normal(0, np.sqrt(1/(n_in)), (n_h, n_in))
    W_rec = np.linalg.qr(np.random.normal(0, 1, (n_h, n_h)))[0]
    W_out = np.random.normal(0, np.sqrt(1/(n_h)), (n_out, n_h))
    b_rec = np.random.normal(0, np.sqrt(1/(n_h)), (n_h,))
    b_out = np.random.normal(0, np.sqrt(1/(n_out)), (n_out,))

    _W_rec = torch.nn.Parameter(torch.tensor(W_rec, requires_grad=True, dtype=torch.float32))
    _W_in = torch.nn.Parameter(torch.tensor(W_in, requires_grad=True, dtype=torch.float32))
    _b_rec = torch.nn.Parameter(torch.tensor(b_rec, requires_grad=True, dtype=torch.float32))
    _W_out = torch.nn.Parameter(torch.tensor(W_out, requires_grad=True, dtype=torch.float32))
    _b_out = torch.nn.Parameter(torch.tensor(b_out, requires_grad=True, dtype=torch.float32))

    return _W_rec, _W_in, _b_rec, _W_out, _b_out

@curry
def rnnTransition(W_in, W_rec, b_rec, activation, alpha, h, x):
    return (1 - alpha) * h + alpha * activation(f.linear(x, W_in, bias=None) + f.linear(h, W_rec, bias=b_rec))

W_rec_, W_in_, b_rec_, _, _ = initializeParametersIO(2, 4, 2)
rnn = rnnTransition(W_in_, W_rec_, b_rec_, f.relu, 1)

# rnn = torch.nn.RNN(2, 4, 1, batch_first=True, nonlinearity='relu')
seq = torch.tensor([[1.0, 1.0], 
                    [3.0, 4.0]])
h0 = torch.ones(4, requires_grad=True)




# test = gradcheck(rnn, (h0, seq[0]), eps=1e-6, atol=1e-4)  requires change to float64 everything
# print(test)



hs = list(scan(rnn, h0, seq))

h0, h1, h2 = hs

def jacobian(_os, _is):
    I_N = torch.eye(len(_os))
    def get_vjp(v):
        return torch.autograd.grad(_os, _is, v)
    return torch.vmap(get_vjp)(I_N)

print(jacobian(h1, h0))


# %%
