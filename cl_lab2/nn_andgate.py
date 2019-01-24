#https://towardsdatascience.com/a-gentle-introduction-to-neural-networks-series-part-1-2b90b87795bc
#https://gist.githubusercontent.com/davidfumo/3005a31a454e4e09108c57ba667b75c5/raw/93672d3763579b951ae66b61aed6a35f8420a05a/single_perceptron.py
from random import choice
import numpy as np
np.random.seed(1)

activation = lambda x: 0 if x < 0 else 1

training_data = [ (np.array([0,0,1]), 0),
                 (np.array([0,1,1]), 0),
                 (np.array([1,0,1]), 0),
                 (np.array([1,1,1]), 1),
                ]

# model parameters
learning_rate = 0.2 
training_steps = 100

# initialize weights 
W = np.random.rand(3) 

for i in range(training_steps):
    
    x, y = choice(training_data) #randomly choose an element from the list
    print(x,y)
    l1 = np.dot(W, x)
    y_pred = activation(l1)
    print(y_pred)
    error = y - y_pred
    update = learning_rate * error * x 
    W += update

# Output after training
print("Predictions after training")
for x, _ in training_data:
    y_pred = np.dot(x, W)
    print("{}: {}".format(x[:2], activation(y_pred)))
