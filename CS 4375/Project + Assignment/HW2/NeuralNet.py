#####################################################################################################################
#   Assignment 2: Neural Network Analysis
#   This is a starter code in Python 3.6 for a neural network.
#   You need to have numpy and pandas installed before running this code.
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it
#       in the README file.
#
#####################################################################################################################


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split # pip install -U scikit-learn
from Network import Network
from activationLayer import ActivationLayer
from FClayer import FClayer
from ctypes import sizeof
from pandas.core.indexers import length_of_indexer


class NeuralNet:
    def __init__(self, dataFile, header=True):
        self.raw_input= pd.read_excel("Real estate valuation data set.xlsx")
        self.raw_input.to_csv ("Test.csv") 


    # TODO: Write code for pre-processing the dataset, which would include
    # standardization, normalization,
    #   categorical to numerical, etc
    def preprocess(self):
        self.processed_data = self.raw_input
        self.processed_data.columns 
        self.processed_data.columns = ['No', 'X1_transaction_date', 'X2_house_age','X3_distance_to_the_nearest_MRT_station','X4_number_of_convenience_stores', 'X5_latitude', 'X6_longitude','Y_house_price_of_unit_area']
        self.processed_data = self.processed_data.drop (columns = ["No"],axis =1)

        list_of_column_names = []
        # Get title of the table
        for row in self.processed_data:
            list_of_column_names.append(row)

        #Fill Na value    
        for i in range(len(list_of_column_names)):
            self.processed_data[list_of_column_names[i]] =  self.processed_data[list_of_column_names[i]].fillna(int( self.processed_data[list_of_column_names[i]].median()))
    
        return self.processed_data


    # TODO: Train and evaluate models for all combinations of parameters
    # specified in the init method. We would like to obtain following outputs:
    #   1. Training Accuracy and Error (Loss) for every model
    #   2. Test Accuracy and Error (Loss) for every model
    #   3. History Curve (Plot of Accuracy against training steps) for all
    #       the models in a single plot. The plot should be color coded i.e.
    #       different color for each model

    def train_evaluate(self):
        ncols = len(self.processed_data.columns)
        nrows = len(self.processed_data.index)
        X = self.processed_data.iloc[:, 0:(ncols - 1)]
        Y = self.processed_data.iloc[:, (ncols-1)]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=0)

        # Seperate your data list, test Data, train Data
        Train_Result = pd.DataFrame(Y_train).to_numpy()
        Train_Data = pd.DataFrame(X_train).to_numpy()
        Test_Result = pd.DataFrame(Y_test).to_numpy()
        Test_Data = pd.DataFrame(X_test).to_numpy()

        n = len(Train_Data[0])
        m = len(Train_Data)

        n1 = len(Test_Data[0])
        m1 = len(Test_Data)

        data_TD = []        # Training Data
        data_TR = []        # Training Result
        data_TestD = []     # Test Data
        data_TestR = []    # Test Result

        for i in range(m):
            temp = []
            temp1 = []
            for j in range(n):
                temp.append(Train_Data[i][j])
            temp1.append(temp)
            data_TD.append(temp1)

        for i in range(m1):
            temp = []
            temp1 = []
            for j in range(n1):
                temp.append(Test_Data[i][j])
            temp1.append(temp)
            data_TestD.append(temp1)
  
        # data_TD = np.array(data_TD)

        for i in range(m):
            temp = []
            temp1 = []
            temp.append(Train_Result[i][0])
            temp1.append(temp)
            data_TR.append(temp1)

        for i in range(m1):
            temp = []
            temp1 = []
            temp.append(Test_Result[i][0])
            temp1.append(temp)
            data_TestR.append(temp1)

        

    
        # Below are the hyperparameters that you need to use for model
        #   evaluation
        activations = ['logistic', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [100, 200] # also known as epochs
        num_hidden_layers = [2, 3]

        # Create the neural network and be sure to keep track of the performance
        #   metrics
        net = Network()
        net.add(FClayer((1,n),(1,3)))
        net.add(ActivationLayer((1,3),(1,3),relu,relu_prime))
        net.add(FClayer((1,3),(1,1)))
        net.add(ActivationLayer((1,1),(1,1),relu,relu_prime))
        net.setup_loss(loss,loss_prime)
        net.fix(np.array(data_TD),np.array(data_TR),learning_rate=0.01,epochs=200)

        out = net.predict([[400,500,1,1,1,1]])
        print(out)

        # Plot the model history for each model in a single plot
        # model history is a plot of accuracy vs number of epochs
        # you may want to create a large sized plot to show multiple lines
        # in a same figure.

        return data_TD, data_TR, data_TestD, data_TestR
   

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

if __name__ == "__main__":
    neural_network = NeuralNet("Real estate valuation data set.xlsx") # put in path to your file
    neural_network.preprocess()
    data_TD, data_TR, data_TestD, data_TestR = neural_network.train_evaluate()
    
    x_train = np.array([[[0.1,0.1,0,1,2,3]],[[400,500,0,1,2,3]],[[1,0,1,1,1,1]],[[1,1,1,3,4,5]],[[1,2,3,5,6,7]]])
    y_train = np.array([[[0]],[[1]],[[1]],[[0]],[[4]]])


    # net = Network()
    # net.add(FClayer((1,6),(1,3)))
    # net.add(ActivationLayer((1,3),(1,3),sigmoid,sigmoid_prime))
    # net.add(FClayer((1,3),(1,1)))
    # net.add(ActivationLayer((1,1),(1,1),sigmoid,sigmoid_prime))
    # net.setup_loss(loss,loss_prime)
    # net.fix(x_train,y_train,learning_rate=0.01,epochs=10)

    # print(data_TR)

    
