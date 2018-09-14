import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  z = X.dot(W)
  max_z = np.max(z, axis = 1)
 
  for i in range(N):
    z_exp = np.exp(z[i, :] - max_z[i])
    sum_exp = np.sum(z_exp)
    loss += -np.log(z_exp[y[i]] / sum_exp)
    prob = z_exp / sum_exp
    prob[y[i]] -= 1
    error = prob.reshape(1, len(prob))
    X_i = X[i, :]
    X_i = X_i.reshape(len(X_i), 1)
    dW += X_i.dot(error)
  loss /= N
  loss += reg * np.sum(W * W)
  dW /= N
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  z = X.dot(W)
  max_z = np.max(z, axis = 1)
  max_z = max_z.reshape(len(max_z), 1)
  z -= max_z
  exp_z = np.exp(z)
  sum_z = np.sum(exp_z, axis = 1)
  sum_z = sum_z[..., np.newaxis]
  softmax_z = exp_z / sum_z
  
  loss = -1/N * (np.sum(np.log(np.choose(y, softmax_z.T)))) + reg * np.sum(W * W)
  softmax_z[range(N), y] -= 1
  dW = X.T.dot(softmax_z) / N
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  ###########################################################################
  return loss, dW

