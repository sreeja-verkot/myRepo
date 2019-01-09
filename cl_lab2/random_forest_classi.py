import pandas as pd  
import numpy as np  
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

print("\nRandom forest classification -predict whether a bank currency note is authentic or not based on four attributes i.e. variance of the image wavelet transformed image, skewness, entropy, and curtosis of the image.\n") 

dataset = pd.read_csv("bill_authentication.csv") 
dataset.head() 

features = dataset.iloc[:, 0:4].values  
labels = dataset.iloc[:, 4].values  

features_train, features_test, labels_train,labels_test = train_test_split(features, labels, test_size=0.2, random_state=0)  

sc=StandardScaler()
features_train = sc.fit_transform(features_train)  
features_test = sc.transform(features_test)  

classifier = RandomForestClassifier(n_estimators=20, random_state=0)  
classifier.fit(features_train, labels_train)  
labels_pred = classifier.predict(features_test)
print(classifier.predict([[1.3114,4.5462,2.2935,0.22541]]))
print("\naccuracy score 0f random foresrt classifier:\n",metrics.accuracy_score(labels_pred, labels_test))  
