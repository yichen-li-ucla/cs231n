from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange

class KNearestNeighbor(object):
    def __init__(self):
      pass

    def train(self, X, y):
      self.X_train = X
      self.y_train = y

    def predict(self, X, k=1, num_loops=0):
      if num_loops == 0:
        dists = self.compute_distances_no_loops(X)
      elif num_loops == 1:
        dists = self.compute_distances_one_loop(X)
      elif num_loops == 2:
        dists = self.compute_distances_two_loops(X)
      else:
        raise ValueError("Invalid value %d for num_loops" % num_loops)
      return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self,X):
      num_test=X.shape[0]
      num_train=self.X_train.shape[0]
      dists=np.zeros((num_test, num_train))
      for i in range(num_test):
        for j in range(num_train):
          dists[i,j]=np.linalg.norm(X[i]-self.X_train[j])
          # XX=X[i]-self.X_train[j]
          # dists[i,j]=np.sqrt(np.dot(XX,XX))
      return dists

    def compute_distances_one_loop(self, X):
      num_test = X.shape[0]
      num_train = self.X_train.shape[0]
      dists = np.zeros((num_test, num_train))
      for i in range(num_test):
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        pass
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
      return dists

    def compute_distances_no_loops(self, X):
      num_test = X.shape[0]
      num_train = self.X_train.shape[0]
      dists=np.sqrt(np.sum(X**2,axis=1,keepdims=True)-2*np.dot(X,self.X_train.T)+np.sum(self.X_train**2,axis=1))
      # NumPy do broadcast with (1,500)+(500,5000)+(5000,)
      return dists

    def predict_labels(self, dists, k=1):
      num_test=dists.shape[0]
      y_pred=np.zeros(num_test)
      from collections import Counter
      for i in range(num_test):
        ind=np.argsort(dists[i])[:k]
        closest_y=self.y_train[ind]
        y_pred[i]=Counter(closest_y).most_common(1)[0][0]
      return y_pred
