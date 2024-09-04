import jax
from utils import *
from jax.random import randint, normal as randn
import jax.numpy as np 

'''
Full batch sampling -> return None, to indicate we use all of the training data
'''
def full_batch_sample():
    def sampler():
        return None
    #
    return sampler
#

'''
Implement minibatch sampling: N is the training dataset size, M is the minibatch size.
Sample M integers from [0,N-1] uniformly at random with replacement
M = 1 gives pure stochastic gradient descent
'''
def minibatch_sampler(N, M):
    def sampler():
        pass
    #
    return sampler
#

'''
Constant step size, specified in step_size_const
'''
def constant_step_size(step_size_const):
    def step_size(t):
        return step_size_const
    #
    return step_size
#

'''
Implement the diminishing step size scheme of Bottou et al. Eq. (4.20)
Specifically, start_step and end_step are, respectively, what the step size at the beginning of optimization, and what the step size should be at the end of optimization, denoted by iteration T
You need to derive the "beta" and "gamma" from the above information, see Bottou et al.
'''
def diminishing_step_size(start_step, end_step, T):
    def step_size(t):
        pass
    #
    return step_size
#

'''
Simple gradient descent update: just return the gradient.
The `descent_direction` function here, and below, takes in (1) the gradient vector, and (2) the data samples (indices, or None) used for computing the gradient
'''
def gd():
    def descent_direction(gt, data_samples):
        return gt
    #
    return descent_direction
#

'''
Implement gradient descent with momentum.
The "momentum" variable represents ... momentum. E.g. when zero, then we revert to standard GD
You will need to update a variable that maintains the (exponentially-weighted) averaged gradient, averaged over optimization
'''
def gd_with_momentum(momentum, D):
    def descent_direction(gt, data_samples):
        pass
    #
    return descent_direction
#

'''
Implement the SAGA method. See Sec. 5.3.2. in Bottou et al. for details.
Assume that data samples is just one sample.
The input `g0` is a list of gradients over all training data items, computed with respect to the weight vector initialization
'''
def saga(g0):
    def descent_direction(gt, data_samples):
        pass
    #
    return descent_direction
#

'''
A generic optimization loop. Expects as input:
    * `w0`: initial weight vector w0
    * `data_sampler`: a function for accessing data samples (called without any arguments) 
    * `loss_func`: a function that computes the loss (NLL for logistic regression), provided a weight vector as input -> this will either be validation loss or training loss
    * `descent_func`: a function for computing a descent direction, given (1) a weight vector, and (2) data samples as input
    * `step_size`: a function for computing the step size, given the current iteration of optimization (a nonnegative integer)
    * `update_method`: Given (1) the gradient, and (2) data samples, returns the vector to use in updating the model parameters
    * `T`: number of optimization steps

During optimization, you should log at each step the loss from `loss_func`. This will be used in the experimental results.

`optimize` should return a 2-tuple:
    * First item is the weight vector upon termination of optimization
    * Second item is an array of losses.
'''
def optimize(w0, data_sampler, loss_func, descent_func, step_size, update_method, T):
    pass
#
