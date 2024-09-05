from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW=np.zeros(W.shape)
    num_classes=W.shape[1]
    num_train=X.shape[0]
    loss=0
    for i in range(num_train):
      scores=X[i].dot(W)
      correct_score=scores[y[i]]
      for j in range(num_classes):
        margin=scores[j]-correct_score+1
        if j!=y[i] and margin>0:
          loss+=margin
          dW[:,j]+=X[i]
          dW[:,y[i]]-=X[i]
    loss/=num_train
    dW/=num_train
    loss+=reg*np.sum(W*W)
    dW+=2*reg*W
    return loss,dW

def svm_loss_vectorized(W, X, y, reg):
    num_classes=W.shape[1]
    num_train=X.shape[0]
    scores=X.dot(W)
    margins=np.maximum(0,scores-scores[np.arange(num_train),y][:,np.newaxis]+1)
    margins[np.arange(num_train),y]=0 # Exclude the correct class
    loss=np.sum(margins)/num_train+reg*np.sum(W*W)
    binary=margins>0
    binary=binary.astype(int)
    binary[np.arange(num_train),y]-=np.sum(binary,axis=1) # Subtract the number of times the margin was violated for the correct class
    dW=X.T.dot(binary)
    dW/=num_train
    dW+=2*reg*W
    return loss,dW
