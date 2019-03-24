#https://www.datacamp.com/community/tutorials/k-means-clustering-python

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
#import seaborn as sns
import matplotlib.pyplot as plt


train = pd.read_csv('/home/user/Desktop/PARSING/myRepo/cl_lab2/train_knnsimple.csv')
test = pd.read_csv('/home/user/Desktop/PARSING/myRepo/cl_lab2/test_knnsimple.csv')
#print("***** Train_Set *****")
#print(train.head())
#print("\n")
#print("***** Test_Set *****")
#print(test.head())

#finding the count of missing data
print("*****In the train set*****")
print(train.isnull().sum())
print("\n")
print("*****In the test set*****")
print(test.isnull().sum())

# Fill missing values with mean column values in the train set
train.fillna(train.mean(), inplace=True)
# Fill missing values with mean column values in the test set
test.fillna(test.mean(), inplace=True)

#finding the count of missing data after imputing .if there is any columns with null data then it will be non numeric
print("*****In the train set*****")
print(train.isnull().sum())
print("\n")
print("*****In the test set*****")
print(test.isnull().sum())

#features like Name, Ticket, Cabin and Embarked do not have any impact on the survival status of the passengers.so remove them.
train = train.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)
test = test.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)

#converting sex to numeric labels
labelEncoder = LabelEncoder()
labelEncoder.fit(train['Sex'])
labelEncoder.fit(test['Sex'])
train['Sex'] = labelEncoder.transform(train['Sex'])
test['Sex'] = labelEncoder.transform(test['Sex'])

X = np.array(train.drop(['Survived'], 1).astype(float))
y = np.array(train['Survived'])

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
kmeans  = KMeans(n_clusters=2, max_iter=600)
kmeans.fit(X_scaled)
correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    print(prediction)
    if prediction[0] == y[i]:
        correct += 1

print("accuracy of kmean:",correct/len(X))


