from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report, confusion_matrix 
import pandas as pd

df = pd.read_csv('/home/user/Desktop/mtech sem1/lab2/lab2/spam.csv', encoding='latin-1')
data_train, data_test, labels_train, labels_test = train_test_split(df.v2,df.v1,test_size=0.2,random_state=0)   
vect = CountVectorizer()
vect.fit(data_train)
X_train_dtm = vect.transform(data_train)
X_test_dtm = vect.transform(data_test)
classifier = KNeighborsClassifier(n_neighbors=3)  
classifier.fit(X_train_dtm, labels_train)
y_pred = classifier.predict(X_test_dtm)
from sklearn import metrics
print("\nAccuracy score of spam classification using knn:\n",metrics.accuracy_score(labels_test, y_pred))
input_msg=set()
msg = input("\nEnter a message\n")
input_msg.add(msg)
print(input_msg)
msg=vect.transform(list(input_msg))
print("message is a:\n",classifier.predict(msg))
print(confusion_matrix(labels_test, y_pred))  
print(classification_report(labels_test, y_pred))    

