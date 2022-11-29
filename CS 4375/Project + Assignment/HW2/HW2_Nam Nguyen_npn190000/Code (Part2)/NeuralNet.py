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
import matplotlib.pyplot as plt
import xlsxwriter



class NeuralNet:
    def __init__(self, dataFile, header=True):
        self.raw_input= pd.read_csv(dataFile,nrows=500)
        self.raw_input.to_csv ("Test.csv") 


    # TODO: Write code for pre-processing the dataset, which would include
    # standardization, normalization,
    #   categorical to numerical, etc
    def preprocess(self):
        self.processed_data = self.raw_input
        self.processed_data.columns 
        self.processed_data.columns = ['1', '2', '3','4','5', '6', '7','8','9']
        self.processed_data = self.processed_data.drop (columns = ["1","9"],axis =1)

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
        data_TestR = []     # Test Result

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
  

        for i in range(m):
            temp = []
            temp1 = []
            temp.append(Train_Result[i][0])
            temp1.append(temp)
            data_TR.append(temp1)

        for i in range(m1):
            data_TestR.append(Test_Result[i][0])
        

    
        # Below are the hyperparameters that you need to use for model
        #   evaluation
        activations = ['logistic', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [100, 200] # also known as epochs
        num_hidden_layers = [2, 3]


        # Create the neural network and be sure to keep track of the performance
        #   metrics
        numberHiddenlayer = 0
        epochs_Num = 0
        learning_rate_options = 0
        i = 0
        error_list = []
        str_list = []
        result_test = []
        error_array = []
        stri = ""
        str1 = ""
        str2 = ""
        for epochs_Num in range (len(max_iterations)):
            for learning_rate_options in range(len(learning_rate)):
                for numberHiddenlayer in range(len( num_hidden_layers)):
                    net = Network()
                    # Input Layer
                    layerActivation(net,n, 5,activations[0])
                    # Hidden Layer
                    if(num_hidden_layers[numberHiddenlayer]==2):
                        layerActivation(net,5, 6,activations[0])
                        layerActivation(net,6, 3,activations[1])

                    if(num_hidden_layers[numberHiddenlayer]==3):
                        layerActivation(net,5, 6,activations[0])
                        layerActivation(net,6, 7,activations[2])
                        layerActivation(net,7, 3,activations[1])
                    # Output layer
                    layerActivation(net,3, 1,activations[0])
                    net.setup_loss(loss,loss_prime)
                    temp = net.fix_update(np.array(data_TD),np.array(data_TR),learning_rate[learning_rate_options],epochs=max_iterations[epochs_Num])
                    error_list.append(temp)
                    stri = ( " num_hidden_layers: " + "% d ") %num_hidden_layers[numberHiddenlayer]
                    str1 = (" learning_rate_options "+"% f ") %learning_rate[learning_rate_options] 
                    str2 = (" max_iterations "+"% d ") %max_iterations[epochs_Num] 
                    stri = stri + str1 + str2
                    str_list.append(stri)
                    i += 1

        result_test = []

        for i in range(m1):
            out = net.predict(data_TestD[i])
            temp = out[0]
            temp1 = temp[0][0]
            result_test.append(temp1)

        diff = differnt(data_TestR,result_test) 
   
        # Plot the model history for each model in a single plot
        # model history is a plot of accuracy vs number of epochs
        # you may want to create a large sized plot to show multiple lines
        # in a same figure.

        n = len(result_test)
        
        rep_num = [i for i in range(n)]
        plt.plot(rep_num,diff)
        plt.title("Display a diagram about Different")
        plt.show()

        rep_num = [i for i in range(n)]
        plt.plot(rep_num,np.array(result_test),label = "line Predict", color = 'black')
        plt.plot(rep_num,np.array(data_TestR),label = "line Real Value", color = 'yellow')
        plt.title("Display a diagram about from your code predict with black line, and real value with blue line")
        plt.show()

        error_array = []
        # Create plot show error of each epochs
        for i in range(len(error_list)):
            e = []
            for j in range(len(error_list[i])):
                out = error_list[i][j]
                e.append(out[0][0])
            error_array.append(e)
        
        for i in range (len(error_array)):
            rep_num = [i for i in range(len(error_array[i]))]
            plt.plot(rep_num,error_array[i])
            plt.title("Display a diagram about MSE Error from each epochs\n"+str_list[i])
            plt.show() 
            plt.close()

        # print out error of each case through excel file
        df = pd.DataFrame(error_array).T
        df.columns 
        df.columns = [str_list[0],str_list[1],str_list[2],str_list[3],str_list[4],str_list[5],str_list[6],str_list[7]]
        df.to_excel(excel_writer = "Folder.xlsx")

# RelU
def relu(x):
    return np.maximum(0,x)
def relu_prime(x):
    x[x<0]=0
    x[x>0]=1
    return x
# sigmoid function
def sigmoid(x):
    return 1.0/(1 + np.exp(-x))
def sigmoid_prime(x):
    temp = (np.exp(-x))/((1 + np.exp(-x))**2)
    return temp
# tanh function
def tanh(x):
    return  (np.exp(x)-np.exp(-x))/(np.exp(x) + np.exp(-x))
def tanh_prime(x):
    temp = 1 - ((np.exp(x)-np.exp(-x))/(np.exp(x) + np.exp(-x)))**2
    return temp
# Cost function C = 0.5(Y^ - Y)^2
def loss(y_actual, y_pre):
    return 0.5*(y_pre - y_actual)**2
def loss_prime(y_actual, y_pre):
    return y_pre - y_actual 
# Define actionvation layer + Neuron layer
def layerActivation(net,input, output,activation):
    if(activation == 'logistic'):    
        net.add(FClayer((1,input),(1,output)))
        net.add(ActivationLayer((1,output),(1,output),sigmoid,sigmoid_prime))
    if(activation == 'tanh'):    
        net.add(FClayer((1,input),(1,output)))
        net.add(ActivationLayer((1,output),(1,output),tanh,tanh_prime))
    if(activation == 'relu'):    
        net.add(FClayer((1,input),(1,output)))
        net.add(ActivationLayer((1,output),(1,output),relu,relu_prime))
# Different of prdict value with actual value
def differnt(dataActual,dataPredict):
    n = len(dataActual)
    diff = []
    for i in range(n):
        error = abs(dataActual[i] - dataPredict[i])
        diff.append(error)
    return diff

if __name__ == "__main__":
    neural_network = NeuralNet("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data") # put in path to your file
    neural_network.preprocess()
    neural_network.train_evaluate()


    
