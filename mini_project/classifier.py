
#Import lib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score 
from sklearn.svm import SVC
from sklearn import metrics
import pandas as pd
df = pd.read_csv('Train.csv')
X=df['TEXT']
Y=df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0) 
vect = CountVectorizer()
vect.fit(X_train)
X_train_dtm = vect.transform(X_train)
X_test_dtm = vect.transform(X_test)
svc=SVC() 

#Default hyperparameters
svc.fit(X_train_dtm,y_train)
y_pred=svc.predict(X_test_dtm)
print('\nAccuracy Score with default hyperparameters:\n')
print(metrics.accuracy_score(y_test,y_pred))
scores = cross_val_score(svc, X, Y, cv=10, scoring='accuracy') #cv is cross validation
print("\ncross validation score:\n",scores)



