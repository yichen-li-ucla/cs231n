from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        self.params['W1']=np.random.normal(0,weight_scale,(input_dim,hidden_dim))
        self.params['W2']=np.random.normal(0,weight_scale,(hidden_dim,num_classes))
        self.params['b1']=np.zeros(hidden_dim)
        self.params['b2']=np.zeros(num_classes)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        W1,b1=self.params['W1'],self.params['b1']
        W2,b2=self.params['W2'],self.params['b2']
        affine1=X.reshape(X.shape[0],-1).dot(W1)+b1
        relu=np.maximum(0,affine1)
        scores=relu.dot(W2)+b2
        if y is None:
            return scores
        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        shift_scores=scores-np.max(scores,axis=1,keepdims=True)
        softmax_output=np.exp(shift_scores)/np.sum(np.exp(shift_scores),axis=1,keepdims=True)
        correct_class_probabilities=softmax_output[np.arange(X.shape[0]), y]
        loss=-np.sum(np.log(correct_class_probabilities))/X.shape[0]
        loss+=0.5*self.reg*(np.sum(W1**2)+np.sum(W2**2))
        dscores=softmax_output
        dscores[np.arange(X.shape[0]),y]-=1
        dscores/=X.shape[0]
        grads['W2']=np.dot(relu.T,dscores)+self.reg*W2
        grads['b2']=np.sum(dscores,axis=0)
        dhidden=np.dot(dscores,W2.T)
        dhidden[relu<=0]=0
        grads['W1']=np.dot(X.reshape(X.shape[0],-1).T,dhidden)+self.reg*W1
        grads['b1']=np.sum(dhidden,axis=0)
        return loss, grads
