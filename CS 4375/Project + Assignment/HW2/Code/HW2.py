from Network import Network
from activationLayer import ActivationLayer
from FClayer import FClayer
import pandas as pd
import numpy as np

# RelU
def relu(x):
  return np.maximum(0,x)
def relu_prime(x):
  x[x<0]=0
  x[x>0]=1
  #print("Temp: ",x)
  return x

# sigmoid function
def sigmoid(x):
  return 1.0/(1 + np.exp(-x))
def sigmoid_prime(x):
  temp = (np.exp(-x))/((1 + np.exp(-x))**2)
  #print("Temp: ",temp)
  return temp
# tanh function

# Cost function C = 0.5(Y^ - Y)^2
def loss(y_actual, y_pre):
  return 0.5*(y_pre - y_actual)**2
def loss_prime(y_actual, y_pre):
  return y_pre - y_actual

x_train = np.array([[[0,0]],[[0,1]],[[1,0]],[[1,1]]])
y_train = np.array([[[0]],[[1]],[[1]],[[0]]])

# net = Network()
# net.add(FClayer((1,2),(1,3)))
# net.add(ActivationLayer((1,3),(1,3),relu,relu_prime))
# net.add(FClayer((1,3),(1,1)))
# net.add(ActivationLayer((1,1),(1,1),relu,relu_prime))
# net.setup_loss(loss,loss_prime)
# net.fix(x_train,y_train,learning_rate=0.01,epochs=10)

net = Network()
net.add(FClayer((1,2),(1,3)))
net.add(ActivationLayer((1,3),(1,3),sigmoid,sigmoid_prime))
net.add(FClayer((1,3),(1,1)))
net.add(ActivationLayer((1,1),(1,1),sigmoid,sigmoid_prime))
net.setup_loss(loss,loss_prime)
net.fix(x_train,y_train,learning_rate=0.01,epochs=100000)

out = net.predict([[0,0]])
print(out)