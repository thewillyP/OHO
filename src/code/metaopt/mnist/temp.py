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
rnn = lambda h, x: rnnTransition(W_in_, W_rec_, b_rec_, lambda x: x, 1, h, x)  #.detach().requires_grad_(True)

# rnn = torch.nn.RNN(2, 4, 1, batch_first=True, nonlinearity='relu')
seq = torch.tensor([[1.0, 1.0], 
                    [3.0, 4.0]])  #.unsqueeze(0)
h0 = torch.ones(4, requires_grad=True)  #.unsqueeze(0)




# test = gradcheck(rnn, (h0, seq[0]), eps=1e-6, atol=1e-4)  requires change to float64 everything
# print(test)

# h1, h2 = list(drop(1)(scan(rnn, h0, seq)))
# h1, h2 = torch.stack((h1.requires_grad_(), h2.requires_grad_()))
hs = list(drop(1)(scan(rnn, h0, seq)))
# hs = torch.stack(hs)
h1 = hs[0]
h2 = hs[1]

# print(h1.requires_grad, h1.grad_fn)
# print(h2.requires_grad, h1.grad_fn)

# h1 = h1.detach().requires_grad_(True)
# h1.register_hook(lambda grad: torch.zeros_like(grad))

# seq = seq.permute(1, 0, 2)
# hs = list(drop(1)(scan(rnn, h0, seq)))
# hs = torch.stack(list(hs)).permute(1, 0, 2)
# h1, h2 = hs[0]

# print(h1)
# print(h2)



# x requires gradient
x = torch.tensor([3.0], requires_grad=True)

# # y is computed, but you can't modify y directly or recompute it
# y = x * 2

# # z is computed using y
# z = x * y

# # Apply a hook on y to zero out its gradient, treating y as constant
# y.register_hook(lambda grad: torch.zeros_like(grad))

# # Backpropagate to compute dz/dx

y = x * 2

# Detach y to treat it as a constant for the next operations
y_detached = y.detach().requires_grad_(True)

# Further operations using the detached tensor
z = y_detached * x


def jacobian(_os, _is):
    I_N = torch.eye(len(_os))
    def get_vjp(v):
        return torch.autograd.grad(_os, _is, v, retain_graph=True, create_graph=True)
    return torch.vmap(get_vjp)(I_N)

# print(h1)
# # print(jacobian(h1, W_rec_))
# aa, = jacobian(h1, W_rec_)
# print(W_rec_.T @ aa+jacobian(h2, W_rec_)[0])
print(jacobian(h2, W_rec_))

"""
1. y = x*2, z = x*y, and I want dz/dx but I want to treat y as a constant. how can I do this
To solve this problem, I will just return a detached, requires grad hidden state every forward pass. This is fine bc BPTT doesn't require me to keep the entire computational graph
2. 
"""


# %%
