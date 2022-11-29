Nam Nguyen
npn190000

Part1: PDF file
Part2: Code

File layer.py
	Class Layer: abstract class or interface class. Try to inform a layer
			Help to get input and output 
			include input shape, and out put shape try to collect matrix size of input and out put
			include forward propogation for a layer
			include back propogation for a layer

File FClayer.py
	Class FClayer(Layer): Full connected layer. Base on layer and have f(x) = w1x + w2x + .... + wnx + bias
			in form process of forward propogation, back propogation (with unpdate function)
			 	

File activationLayer.py
	Class ActivationLayer(Layer): Activation layer try to be inform Sigmoid (Logistic) function, or Tanh function, or Relu function
			in form process of forward propogation, back propogation (with unpdate function)

File Network.py
	Class Network: Create a network by add function, we can add FClayer or ActivationLayer. It depend on programmer.
			include predict process
			fix_update function process to update weight, by learning rate, and epochs
				Show forward propagation process, and  Backward propagation
				Can have applity save error - MSE

File NeuralNet.py
	Class NeuralNet: Train data set 
	Runner file
	Using command: python NeuralNet.py

File Test.csv: data from UCI ML
FIle Folder.xlsx: data collect from running program to get error each epoch. Output for your dataset summarized in a tabular format for different
combination of parameters.
