import pandas as pd
import numpy as np

import math
import matplotlib.pyplot as plt #pip install openpyxl
import io
from sklearn.model_selection import train_test_split

# Working with data base from UCI
dataFrame = pd.read_excel("https://archive.ics.uci.edu/ml/machine-learning-databases/00477/Real%20estate%20valuation%20data%20set.xlsx")
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


# Predict function for a new_radio. Find Y^
def predict(new_Data, weight_list, bias):
    predict = bias
    for i in range(len(new_Data)):
      predict += new_Data[i]*weight_list[i] 
    return predict

# Cost function. include initial X,Y, current weight,bias. MSE
def cost_function(trainData,trainResult,weight_list,bias):
  n = len(trainData[0])
  m = len(trainData)
  sum_error = 0
  pre = 0.0

  for i in range(m):
    pre = bias
    for j in range(n):
      pre = pre + weight_list[j]*trainData[i][j]
    sum_error += (trainResult[i] - (pre))**2
  return sum_error/m

# RSS
def RSS(trainData,trainResult,weight_list,bias):
  m = len(trainData)
  return cost_function(trainData,trainResult,weight_list,bias)*m

# TSS
def TSS(trainData,trainResult,weight_list,bias):
  n = len(trainData[0])
  m = len(trainData)
  sum_error = 0
  pre = 0.0

  pre = np.median(np.array(trainData)) 
  for i in range(m):
    sum_error += (trainResult[i] - (pre))**2
  
  return sum_error/m
#R^2
def R2(trainData,trainResult,weight_list,bias):
  temp1 = RSS(trainData,trainResult,weight_list,bias)
  temp2 = TSS(trainData,trainResult,weight_list,bias)

  return 1 - (temp1/temp2)

#Create new weight and new bias use Gradient Descent
# Use learning rate
def update(trainData,testData,weight_list,bias,learning_rate):
  n = len(trainData[0])
  m = len(trainData)
  pre = 0
  weight_temp = []
  bias_temp = 0.0
  
  for j in range(n):
    weight_temp.append(0.0)
 
  for i in range(m):
    pre = bias
    for j in range(n):
        pre += weight_list[j]*trainData[i][j]
    for j in range(n):
          weight_temp[j] += -1*trainData[i][j]*(testData[i]-pre)
     
    bias_temp += -1*(testData[i]-pre) 

  for j in range(n):
    weight_list[j] = weight_list[j]- ((weight_temp[j]*2)/m)*learning_rate
  e =  ((bias_temp*2)/m)*learning_rate
  bias =  bias - e 

  return weight_list,float(bias)

# Training function
def train(trainData,testData,weight_list,bias,learning_rate,iter):
  cost_his = []
  r2_his = []
  r2 = []
  for i in range(iter):
      weight_list,bias = update(trainData,testData,weight_list,bias,learning_rate) # update weight , bias
      
      cost = cost_function(trainData,testData,weight_list,bias) # count MSE
      r2 = R2(trainData,testData,weight_list,bias) # R^2
      cost_his.append(cost)
      r2_his.append(r2)
  return bias,weight_list,cost_his,r2_his

# Set a number for_loop to repeat the training
n = 400

# Create initial weigh list
weight_list = []
for j in range(len(Train_Data[0])):
    weight_list.append(0.1)

# Testting data
bias,weight_list,cost,r_2 = train(Train_Data,Train_Result,weight_list,0.1,0.00000001,n)

# Display result 

print("Y_interception: ",bias)
print("Weight list: \n", weight_list)
print("MSE list \n",cost)
print("R power 2 list \n",r_2)

print("\nDisplay a diagram about MSE (cost)\n")
rep_num = [i for i in range(n)]
plt.plot(rep_num,cost)
plt.show()
print("\nDisplay a diagram about R**2\n")
rep_num = [i for i in range(n)]
plt.plot(rep_num,r_2)
plt.show()

# Testing with data
Y_predict = [] 
length_of_Test_List = len(Test_Data)

for i in range(length_of_Test_List):
  Y_predict.append(predict(Test_Data[i],weight_list,bias))

# Other Diagram

print("\nDisplay a diagram about from your code predict with black line, and real value with blue line\n")
rep_num = [i for i in range(length_of_Test_List)]
plt.plot(rep_num,np.array(Y_predict),label = "line Predict", color = 'black')
plt.plot(rep_num,np.array(Test_Result),label = "line Real Value", color = 'blue')
plt.show()
#