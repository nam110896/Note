from layer import Layer
import pandas as pd
import numpy as np

# Activation Layer
class ActivationLayer(Layer):
  def __init__(self,input_shape,output_shape,active_configuration,activation_prime):
    self.input_shape = input_shape      # Example matrix (1,4)
    self.output_shape = output_shape    # Example matrix depend
    self.activation = active_configuration
    self.activation_prime = activation_prime
  
  def forward_propagation(self, input):
    self.input = input
    self.output = self.activation(input)
    return self.output
  def backward_propagation(self, output_error,learning_rate):
    return self.activation_prime(self.input)*output_error