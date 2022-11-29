from FClayer import FClayer
from activationLayer import ActivationLayer

#Network
class Network:
  def __init__(self):
    self.layers = []
    self.loss = None
    self.loss_prime = None

  def add(self,layer):
    self.layers.append(layer)

  def setup_loss(self,loss,loss_prime):
    self.loss = loss
    self.loss_prime = loss_prime

  # Predict for each layer forward_propagation 
  def predict(self,input):
  #   # input example matrix [(1,3),(3,5),(3,4)]
    n = len(input)
    result = []
    for i in range(n):
      output = input[i] # Example output matrix (1.3)
      for layer in self.layers:
        output = layer.forward_propagation(output)
      result.append(output)
    return result

  # Upadate weights and bias
  def fix_update(self,x_train,y_train,learning_rate,epochs):
    err_list = []
    # Check data set
    n = len(x_train)
    for i in range(epochs):
      err = 0
      for j in range(n):
        # forward propagation
        output = x_train[j]
        for layer in self.layers:
          output = layer.forward_propagation(output)
        # Error of each layer MSE
        err += self.loss(y_train[j],output)
        error = self.loss_prime(y_train[j],output)

        # Backward propagation
        for layer in reversed(self.layers):
          error = layer.backward_propagation(error,learning_rate)
      err = err/n
      err_list.append(err)
      # print ('epoch : %d/%d err = %f'%(i,epochs,err))
    return err_list