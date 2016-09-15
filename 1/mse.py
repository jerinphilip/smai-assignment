
# MSE Methods
# Minimize e^2, e = |Ya - b|
import numpy as np
import numpy.linalg as la

def closed(ys, b):
   y_pseudo_inverse = la.pinv(ys)
   #print(y_pseudo_inverse.dot(b.T))
   return y_pseudo_inverse.dot(b.T)


