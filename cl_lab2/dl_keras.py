# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
#excluding first 2 columns as they are useless
X = dataset.iloc[:, 3:13].values
#target values
y = dataset.iloc[:, 13].values
#ncode strings into numeric 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
#LabelEncoder has replaced France with 0, Germany 1 and Spain 2 but Germany is not higher than France and France is not smaller than Spain so we need to create a dummy variable for Country.  We don’t need to do same for Gender Variable as it is binary.
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
#Initializing Neural Network
classifier = Sequential()

#Our first parameter is output_dim. It is simply the number of nodes you want to add to this layer. init is the initialization of Stochastic Gradient Decent. In Neural Network we need to assign weights to each mode which is nothing but importance of that node. At the time of initialization, weights should be close to 0 and we will randomly initialize weights using uniform function. input_dim parameter is needed only for first layer as model doesn’t know the number of our input variables. Remember in our case, the total number of input variables are 11. In the second layer model automatically knows the number of input variable from the first hidden layer.Activation Function: Very important to understand. Neuron applies activation function to weighted sum(summation of Wi * Xi where w is weight, X is input variable and i is suffix of W and X). The closer the activation function value to 1 the more activated is the neuron and more the neuron passes the signal. Which activation function should be used is critical task. Here we are using rectifier(relu) function in our hidden layer and Sigmoid function in our output layer as we want binary result from output layer but if the number of categories in output layer is more than 2 then use SoftMax function.

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#First argument is Optimizer, this is nothing but the algorithm you wanna use to find optimal set of weights(Note that in step 9 we just initialized weights now we are applying some sort of algorithm which will optimize weights in turn making out neural network more powerful. This algorithm is Stochastic Gradient descent(SGD). Among several types of SGD algorithm the one which we will use is ‘Adam’. If you go in deeper detail of SGD, you will find that SGD depends on loss thus our second parameter is loss. Since out dependent variable is binary, we will have to use logarithmic loss function called ‘binary_crossentropy’, if our dependent variable has more than 2 categories in output then use ‘categorical_crossentropy’. We want to improve performance of our neural network based on accuracy so add metrics as accuracy

# Compiling Neural Network
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting our model  Batch size is used to specify the number of observation after which you want to update weight. Epoch is nothing but the total number of iterations. Choosing the value of batch size and epoch is trial and error there is no specific rule for that.
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Creating the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
