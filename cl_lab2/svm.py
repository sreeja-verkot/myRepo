#https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/
#https://www.kaggle.com/nirajvermafcb/support-vector-machine-detail-analysis/notebook

#gender recognition using voice details
import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score

df = pd.read_csv('voice.csv')
df.head()
print("Total number of labels: {}\n".format(df.shape[0]))
print("Number of male: {}\n".format(df[df.label == 'male'].shape[0]))
print("Number of female: {}\n".format(df[df.label == 'female'].shape[0]))

X=df.iloc[:, :-1]
X.head()
from sklearn.preprocessing import LabelEncoder
y=df.iloc[:,-1]

# Encode label category
# male -> 1
# female -> 0

gender_encoder = LabelEncoder()
y = gender_encoder.fit_transform(y)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
from sklearn.svm import SVC
from sklearn import metrics
svc=SVC() 

#Default hyperparameters
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
print('\nAccuracy Score with default hyperparameters:\n')
print(metrics.accuracy_score(y_test,y_pred))
scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy') #cv is cross validation
print("\ncross validation score:\n",scores)

svc=SVC(kernel='linear')
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
print('\nAccuracy Score with linear kernel:\n')
print(metrics.accuracy_score(y_test,y_pred))
scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy') #cv is cross validation
print("\ncross validation score:\n",scores)

svc=SVC(kernel='rbf')
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
print('\nAccuracy Score with rbf kernel:\n')
print(metrics.accuracy_score(y_test,y_pred))
scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy') #cv is cross validation
print("\ncross validation score:\n",scores)

svc=SVC(kernel='poly')
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
print('\nAccuracy Score with polynomial kernal:\n')
print(metrics.accuracy_score(y_test,y_pred))
scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy') #cv is cross validation
print("\ncross validation score:\n",scores)

