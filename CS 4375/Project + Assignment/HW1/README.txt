Nam Nguyen - npn190000

This program run good with Google Colab. 

Part1:

Let Y = w1x1 + w2x2 + w3x3 + ... + wnxn + bias

X = {x1,x2,x3,...,xn} 
w = {w1, w2, ..., wn}
bias = Y intercept

In this program:

At begin is some step to solve data from UCI machine leaning
Function

# Predict function for a new_radio. Find Y^
def predict(new_Data, weight_list, bias):
 	-> in predict funtion we need to have an array with enough attribute, and weight_ list, and Y intercept to get a new predict result.

# Cost function. include initial X,Y, current weight,bias. MSE
def cost_function(trainData,trainResult,weight_list,bias):
	-> Use the equation of MSE. We will need data for train (trainData), the result data list(trainResult),and weight_ list, and Y intercept
To apply to the equation. 

# RSS
def RSS(trainData,trainResult,weight_list,bias):
	-> Use the equation of RSS. We will need data for train (trainData), the result data list(trainResult),and weight_ list, and Y intercept
To apply to the equation. 

# TSS
def TSS(trainData,trainResult,weight_list,bias):
	-> Use the equation of TSS. We will need data for train (trainData), the result data list(trainResult),and weight_ list, and Y intercept
To apply to the equation. 

#R^2
def R2(trainData,trainResult,weight_list,bias):
	-> Use the equation of R^2. We will need data for train (trainData), the result data list(trainResult),and weight_ list, and Y intercept
To apply to the equation. 

#Create new weight and new bias use Gradient Descent
# Use learning rate
def update(trainData,testData,weight_list,bias,learning_rate):
	-> Update new weight list (weight_list), update new bias (Y intercept). By using a specific learning rate (Example 0,00001).
Then using equation of dl/dw, and dl/d(Y intercept).

# Training function
def train(trainData,testData,weight_list,bias,learning_rate,iter):
	-> Base on function above, We will need data for train (trainData), the result data list(trainResult),and weight_ list, and Y intercept
To train our program we need iter number that tell your for_loop to repeat the training process.


Example testting:
. bias,weight_list,cost,r_2 = train(Train_Data,Train_Result,weight_list,0.1,0.00000001,n)
-> Train by using liner regression using Gradient descent to get new weight_list, and Y_intercept.

Diagram:
rep_num = [i for i in range(n)]
plt.plot(rep_num,cost)
plt.show()

-> Show diagram, by using list of cost (MSE) to display the result.

rep_num = [i for i in range(n)]
plt.plot(rep_num,r_2)
plt.show()

-> Show diagram, by using list of cost (MSE) to display the result.


Part 2:

Using from sklearn import linear_model to predict.

print("\nDisplay a diagram about from Sklearn code predict with yellow line, and real value with blue line\n")
rep_num = [i for i in range(length_of_Test_List)]
plt.plot(rep_num,np.array(Y_sklearn),label = "line Predict from Sklearn", color = 'yellow')
plt.plot(rep_num,np.array(Test_Result),label = "line Real Value", color = 'blue')
plt.show()


print("\nDisplay a diagram about from Sklearn code predict with yellow line, and your code predict value with black line\n")
rep_num = [i for i in range(length_of_Test_List)]
plt.plot(rep_num,np.array(Y_sklearn),label = "line Predict from Sklearn", color = 'yellow')
plt.plot(rep_num,np.array(Y_predict), label = "line Predict", color = 'black')
plt.show()