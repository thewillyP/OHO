import os, sys, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from copy import copy
import numpy as np
from metaopt.util_ml import *
from metaopt.util import *
from toolz import accumulate
from myfunc import jacobian

def split_weight_matrix(A, sizes, axis=1):
    """Splits a weight matrix along the specified axis (0 for row, 1 for
    column) into a list of sub arrays of size specified by 'sizes'."""

    idx = [0] + np.cumsum(sizes).tolist()
    if axis == 1:
        ret = [np.squeeze(A[:,idx[i]:idx[i+1]]) for i in range(len(idx) - 1)]
    elif axis == 0:
        ret = [np.squeeze(A[idx[i]:idx[i+1],:]) for i in range(len(idx) - 1)]
    return ret

def norm(z):
    """Computes the L2 norm of a numpy array."""

    return np.sqrt(np.sum(np.square(z)))


class Learning_Algorithm:
    """Parent class for all learning algorithms.

    Attributes:
        rnn (network.RNN): An instance of RNN to be trained by the network.
        n_* (int): Extra pointer to rnn.n_* (in, h, out) for conveneince.
        m (int): Number of recurrent "input dimensions" n_h + n_in + 1 including
            task inputs and constant 1 for bias.
        q (numpy array): Array of immediate error signals for the hidden units,
            i.e. the derivative of the current loss with respect to rnn.a, of
            shape (n_h).
        W_FB (numpy array or None): A fixed set of weights that may be provided
            for an approximate calculation of q in the manner of feedback
            alignment (Lillicrap et al. 2016).
        L2_reg (float or None): Strength of L2 regularization parameter on the
            network weights.
        a_ (numpy array): Array of shape (n_h + 1) that is the concatenation of
            the network's state and the constant 1, used to calculate the output
            errors.
        q (numpy array): The immediate loss derivative of the network state
            dL/da, calculated by propagate_feedback_to_hidden.
        q_prev (numpy array): The q value from the previous time step."""

    def __init__(self, rnn, allowed_kwargs_=set(), **kwargs):
        """Initializes an instance of learning algorithm by specifying the
        network to be trained, custom allowable kwargs, and kwargs.

        Args:
            rnn (network.RNN): An instance of RNN to be trained by the network.
            allowed_kwargs_ (set): Set of allowed kwargs in addition to those
                common to all child classes of Learning_Algorithm, 'W_FB' and
                'L2_reg'."""

        allowed_kwargs = {'W_FB', 'L1_reg', 'L2_reg', 'CL_method',
                          'maintain_sparsity', 'sigma'}.union(allowed_kwargs_)

        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument passed'
                                'to Learning_Algorithm.__init__: ' + str(k))

        # Set all non-specified kwargs to None
        for attr in allowed_kwargs:
            if not hasattr(self, attr):
                setattr(self, attr, None)

        # Make kwargs attributes of the instance
        self.__dict__.update(kwargs)

        # Define basic learning algorithm properties
        self.rnn = rnn
        self.n_in = self.rnn.n_in
        self.n_h = self.rnn.n_h
        self.n_out = self.rnn.n_out
        self.m = self.n_h + self.n_in + 1
        self.q = np.zeros(self.n_h)

    def get_outer_grads(self):
        """Calculates the derivative of the loss with respect to the output
        parameters rnn.W_out and rnn.b_out.

        Calculates the outer gradients in the manner of a perceptron derivative
        by taking the outer product of the error with the "regressors" onto the
        output (the hidden state and constant 1).

        Returns:
            A numpy array of shape (rnn.n_out, self.n_h + 1) containing the
                concatenation (along column axis) of the derivative of the loss
                w.r.t. rnn.W_out and w.r.t. rnn.b_out."""

        self.a_ = np.concatenate([self.rnn.a, np.array([1])])
        return np.multiply.outer(self.rnn.error, self.a_)

    def propagate_feedback_to_hidden(self):
        """Performs one step of backpropagation from the outer-layer errors to
        the hidden state.

        Calculates the immediate derivative of the loss with respect to the
        hidden state rnn.a. By default, this is done by taking rnn.error (dL/dz)
        and applying the chain rule, i.e. taking its matrix product with the
        derivative dz/da, which is rnn.W_out. Alternatively, if 'W_FB' attr is
        provided to the instance, then these feedback weights, rather the W_out,
        are used, as in feedback alignment. (See Lillicrap et al. 2016.)

        Updates q to the current value of dL/da."""

        self.q_prev = np.copy(self.q)

        if self.W_FB is None:
            self.q = self.rnn.error.dot(self.rnn.W_out)
        else:
            self.q = self.rnn.error.dot(self.W_FB)

    def L2_regularization(self, grads):
        """Adds L2 regularization to the gradient.

        Args:
            grads (list): List of numpy arrays representing gradients before L2
                regularization is applied.
        Returns:
            A new list of grads with L2 regularization applied."""

        # Get parameters affected by L2 regularization
        L2_params = [self.rnn.params[i] for i in self.rnn.L2_indices]
        # Add to each grad the corresponding weight's current value, weighted
        # by the L2_reg hyperparameter.
        for i_L2, W in zip(self.rnn.L2_indices, L2_params):
            grads[i_L2] += self.L2_reg * W
        # Calculate L2 loss for monitoring purposes
        self.L2_loss = 0.5 * sum([norm(p) ** 2 for p in L2_params])
        return grads

    def L1_regularization(self, grads):
        """Adds L1 regularization to the gradient.

        Args:
            grads (list): List of numpy arrays representing gradients before L1
                regularization is applied.
        Returns:
            A new list of grads with L1 regularization applied."""

        # Get parameters affected by L1 regularization
        # (identical as those affected by L2)
        L1_params = [self.rnn.params[i] for i in self.rnn.L2_indices]
        # Add to each grad the sign of the corresponding parameter weighted
        # by L1 reg strength
        for i_L1, W in zip(self.rnn.L2_indices, L1_params):
            grads[i_L1] += self.L1_reg * np.sign(W)
        # Calculate L2 loss for monitoring purposes
        self.L1_loss = sum([norm(p) for p in L1_params])
        return grads

    def apply_sparsity_to_grads(self, grads):
        """"If called, modifies gradient to make 0 any parameters that are
        already 0 (only for L2 params).

        Args:
            grads (list): List of numpy arrays representing gradients before L2
                regularization is applied.
        Returns:
            A new list of grads with L2 regularization applied."""

        # Get parameters affected by L2 regularization
        L2_params = [self.rnn.params[i] for i in self.rnn.L2_indices]
        # AMultiply each gradient by 0 if it the corresponding weight is
        # already 0
        for i_L2, W in zip(self.rnn.L2_indices, L2_params):
            grads[i_L2] *= (W != 0)
        return grads

    def add_noise_to_grads(self, grads):
        """If called, modifies the gradient by adding in gaussian noise with
        standard deviation specified by self.sigma."""

        for i_grad, shape in enumerate(self.rnn.shapes):
            grads[i_grad] += np.random.normal(0, self.sigma, shape)
        return grads

    def __call__(self):
        """Calculates the final list of grads for this time step.

        Assumes the user has already called self.update_learning_vars, a
        method specific to each child class of Real_Time_Learning_Algorithm
        that updates internal learning variables, e.g. the influence matrix of
        RTRL. Then calculates the outer grads (gradients of W_out and b_out),
        updates q using propagate_feedback_to_hidden, and finally calling the
        get_rec_grads method (specific to each child class) to get the gradients
        of W_rec, W_in, and b_rec as one numpy array with shape (n_h, m). Then
        these gradients are split along the column axis into a list of 5
        gradients for W_rec, W_in, b_rec, W_out, b_out. L2 regularization is
        applied if L2_reg parameter is not None.

        Returns:
            List of gradients for W_rec, W_in, b_rec, W_out, b_out."""

        self.outer_grads = self.get_outer_grads()
        self.propagate_feedback_to_hidden()
        self.rec_grads = self.get_rec_grads()
        rec_grads_list = split_weight_matrix(self.rec_grads,
                                             [self.n_h, self.n_in, 1])
        outer_grads_list = split_weight_matrix(self.outer_grads,
                                               [self.n_h, 1])
        grads_list = rec_grads_list + outer_grads_list

        if self.L1_reg is not None:
            grads_list = self.L1_regularization(grads_list)

        if self.L2_reg is not None:
            grads_list = self.L2_regularization(grads_list)

        if self.CL_method is not None:
            grads_list = self.CL_method(grads_list)

        if self.maintain_sparsity:
            grads_list = self.apply_sparsity_to_grads(grads_list)

        if self.sigma is not None:
            grads_list = self.add_noise_to_grads(grads_list)

        return grads_list

    def reset_learning(self):
        """Resets internal variables of the learning algorithm (relevant if
        simulation includes a trial structure). Default is to do nothing."""

        pass


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

    def __init__(self, n_in: int, n_h: int, n_out: int, lr_init, lambda_l2, is_cuda=0, M_decay=1):
        """Inits an RTRL instance by setting the initial dadw matrix to zero."""

        self.dadw = torch.zeros((n_h, n_h))
        self.M_decay = M_decay



        self.rnn = nn.RNN(n_in, n_h, 1, batch_first=True, nonlinearity='tanh')  # sampels weights from uniform which is pretty big
        self.fc = nn.Linear(n_h, n_out)

        self.initH = lambda x: torch.zeros(1, x.size(0), n_h).to('cpu' if is_cuda==0 else 'gpu') 
        # self.reshapeImage = lambda images: images.view(-1, sequence_length, n_in).to('cpu' if is_cuda==0 else 'gpu')


        param_sizes = [p.numel() for p in self.parameters()]
        self.n_params = sum(param_sizes)
        self.param_shapes = [tuple(p.shape) for p in self.parameters()]
        self.param_cumsum = np.cumsum([0] + param_sizes)

        self.reset_jacob(is_cuda)
        self.eta  = lr_init
        self.lambda_l2 = lambda_l2
        self.grad_norm = 0
        self.grad_norm_vl = 0
        self.grad_angle = 0
        self.param_norm = 0
        self.dFdlr_norm = 0
        self.dFdl2_nrom = 0
    
    def forward(self, x, logsoftmaxF=1):
        h0 = self.initH(x)
        x, _ = self.rnn(x, h0)
        x = self.fc(x)
        return x


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



# Inherit from Function
class LinearFunction(torch.autograd.Function):

    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(input, weight, bias):
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx, inputs, output):
        input, weight, bias = inputs
        ctx.save_for_backward(input, weight, bias)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias




class RTRLFunction(torch.autograd.Function):
    @staticmethod
    def forward(h0, seq, rnn, state):
        new_hidden_state = rnn(h0, seq)
        return new_hidden_state
    
    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx, inputs, output):
        seq, h0, _, state = inputs
        ctx.save_for_backward(seq, h0, output)
        ctx.state = state

# The gradient of this is literally the new influenceMatrix which I can pass in again

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        influenceMatrix = ctx.state.influenceMatrix
        seq, h0, hs = ctx.saved_tensors

        n = seq.size(1)
        for i in range(n):
            xs = seq[:,i,:]
        
        def fn(state, ht):
            infMtx, ht_1 = state
            dActivation = jacobian(ht, ht_1)
            dActivation @ infMtx

        accumulate(fn, hs, (influenceMatrix, h0))
        
        
        # Compute gradient of the hidden state wrt to the weight (RTRL update)
        # Partial derivatives of hidden state wrt weight
        d_hidden_d_weight = (1 - new_hidden_state ** 2).unsqueeze(-1) * input_tensor.unsqueeze(-2)
        
        # Compute gradient for the weight
        grad_weight = torch.matmul(d_hidden_d_weight.transpose(0, 1), grad_output)
        
        # Compute gradient for the input (for upstream layers)
        grad_input = torch.matmul(grad_output, weight.T)
        
        # Compute gradient for the hidden state (previous timestep)
        grad_hidden = grad_output * (1 - new_hidden_state ** 2)

        return grad_input, grad_hidden, grad_weight

# Example usage:

