from layer import Layer
import pandas as pd
import numpy as np

#Full Conected Layer
class FClayer(Layer):
  def __init__(self,input_shape,output_shape):
    self.input_shape = input_shape      # Example matrix (1,3)
    self.output_shape = output_shape    # Example matrix (1,4)
    
    self.weights = np.random.rand(input_shape[1],output_shape[1]) -0.5    # Example matrix (3,4)
    self.bias = np.random.rand(1, output_shape[1]) -0.5                   # Example matrix (1,4)

  def forward_propagation(self, input):
    self.input = input
    self.output = np.dot(self.input,self.weights) + self.bias
    return self.output

  def backward_propagation(self, output_error,learning_rate):
    currentLayer_error = np.dot(output_error,self.weights.T)    # Example matrix (1,4) * matrix (4,3) = matrix (1,3)
    
    # Update weights by using Gradient Desent
    dw = np.dot(self.input.T,output_error)                      # Example matrix(3,1) * mtrix (1,4) = matrix (3,4)
    self.weights -= dw*learning_rate
    self.bias -= output_error*learning_rate

    return currentLayer_error    