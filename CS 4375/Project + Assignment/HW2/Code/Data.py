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
  if(x > -200).all():
    if (x < 200).all():
      return 1.0/(1 + np.exp(-x))
  if (x >-114 ) .any():
     if(x < 114).any():
        
  if(x >= 114).any(): return 1
  if(x <= 114).any(): return 0
  
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

# x_train = np.array([[[0,0]],[[0,1]],[[1,0]],[[1,1]]])
# y_train = np.array([[[0]],[[1]],[[1]],[[0]]])

# net = Network()
# net.add(FClayer((1,2),(1,3)))
# net.add(ActivationLayer((1,3),(1,3),relu,relu_prime))
# net.add(FClayer((1,3),(1,1)))
# net.add(ActivationLayer((1,1),(1,1),relu,relu_prime))
# net.setup_loss(loss,loss_prime)
# net.fix(x_train,y_train,learning_rate=0.01,epochs=10)

# net = Network()
# net.add(FClayer((1,2),(1,3)))
# net.add(ActivationLayer((1,3),(1,3),sigmoid,sigmoid_prime))
# net.add(FClayer((1,3),(1,1)))
# net.add(ActivationLayer((1,1),(1,1),sigmoid,sigmoid_prime))
# net.setup_loss(loss,loss_prime)
# net.fix(x_train,y_train,learning_rate=0.01,epochs=100000)

# out = net.predict([[0,0]])
# print(out)

import io
from sklearn.model_selection import train_test_split

# Working with data base from UCI
dataFrame = pd.read_excel("Real estate valuation data set.xlsx")
dataFrame.to_csv ("Test.csv")
dataFrame.columns 
dataFrame.columns = ['No', 'X1_transaction_date', 'X2_house_age','X3_distance_to_the_nearest_MRT_station','X4_number_of_convenience_stores', 'X5_latitude', 'X6_longitude','Y_house_price_of_unit_area']
df = dataFrame.drop (columns = ["No"],axis =1)
train, test = train_test_split(df, test_size=0.2)
X = pd.DataFrame(np.c_[df['X1_transaction_date'], df['X2_house_age'], df['X3_distance_to_the_nearest_MRT_station'],df['X4_number_of_convenience_stores'],df['X5_latitude'],df['X6_longitude']], 
                 columns = ['X1_transaction_date', 'X2_house_age','X3_distance_to_the_nearest_MRT_station','X4_number_of_convenience_stores', 'X5_latitude', 'X6_longitude'])
Y = df['Y_house_price_of_unit_area']

# Split 80:20
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)
# print(X_train)
# print(X_test)
# print(Y_train)
# print(Y_test)

from ctypes import sizeof
from pandas.core.indexers import length_of_indexer
list_of_column_names = []
# Get title of the table
for row in df:
    list_of_column_names.append(row)

#Fill Na value    
for i in range(len(list_of_column_names)):
    df[list_of_column_names[i]] =  df[list_of_column_names[i]].fillna(int( df[list_of_column_names[i]].median()))

# Seperate your data list, test Data, train Data
Train_Result = pd.DataFrame(Y_train).to_numpy()
Train_Data = pd.DataFrame(X_train).to_numpy()
Test_Result = pd.DataFrame(Y_test).to_numpy()
Test_Data = pd.DataFrame(X_test).to_numpy()

n = len(Train_Data[0])
m = len(Train_Data)



print(m,n)# 336 6
data_TD = []  # Training Data
data_TR = []  # Training Result

for i in range(m):
  temp = []
  temp1 = []
  for j in range(n):
    temp.append(Train_Data[i][j])
  temp1.append(temp)
  data_TD.append(temp1)
  
# data_TD = np.array(data_TD)

for i in range(m):
  temp = []
  temp1 = []
  temp.append(Train_Result[i][0])
  temp1.append(temp)
  data_TR.append(temp1)

# for i in range(m):
#   print(data_TR[i], "\n")

net = Network()
net.add(FClayer((1,len(Train_Data[0])),(1,3)))
net.add(ActivationLayer((1,3),(1,3),sigmoid,sigmoid_prime))
net.add(FClayer((1,3),(1,1)))
net.add(ActivationLayer((1,1),(1,1),sigmoid,sigmoid_prime))
net.setup_loss(loss,loss_prime)
net.fix(np.array(data_TD),np.array(data_TR),learning_rate=0.01,epochs=1)

# x_train = np.array([[[0.1,0.1,0,1,2,3]],[[400,500,0,1,2,3]],[[1,0,1,1,1,1]],[[1,1,1,3,4,5]],[[1,2,3,5,6,7]]])
y_train = np.array([[[0]],[[1]],[[1]],[[0]],[[4]]])


# net = Network()
# net.add(FClayer((1,len(Train_Data[0])),(1,3)))
# net.add(ActivationLayer((1,3),(1,3),sigmoid,sigmoid_prime))
# net.add(FClayer((1,3),(1,1)))
# net.add(ActivationLayer((1,1),(1,1),sigmoid,sigmoid_prime))
# net.setup_loss(loss,loss_prime)
# net.fix(x_train,y_train,learning_rate=0.01,epochs=10)

out = net.predict([[0,0,1,1,1,1]])
print(out)

