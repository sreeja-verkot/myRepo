#https://www.xoriant.com/blog/product-engineering/decision-trees-machine-learning-algorithm.html

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_iris()

#Extracting data attributes
X = data.data
### Extracting target/ class labels
y = data.target

#Using the train_test_split to create train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 47, test_size = 0.25)

#set the 'criterion' to 'entropy', which sets the measure for splitting the attribute to information gain
clf = DecisionTreeClassifier(criterion = 'entropy')

clf.fit(X_train, y_train)
y_pred =  clf.predict(X_test)
print('\nAccuracy Score on test data:\n ', accuracy_score(y_test,y_pred))
print(clf.predict([[5.0,2.0,3.5,1.0]]))
