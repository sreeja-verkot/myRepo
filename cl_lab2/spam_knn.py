from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report, confusion_matrix 
import pandas as pd
import nltk
import string
df = pd.read_csv('/home/user/Desktop/mtech sem1/lab2/lab2/spam.csv', encoding='latin-1')
#print (df.head())
data_train, data_test, labels_train, labels_test = train_test_split(df.v2,df.v1,test_size=0.2,random_state=0)   
#print (data_train[:10])
tfidf_vectorizer = TfidfVectorizer()
X_train = tfidf_vectorizer.fit_transform(data_train)
X_pred=tfidf_vectorizer.fit_transform(labels_train)
print(type(X_train))
classifier = KNeighborsClassifier(n_neighbors=5)  
classifier.fit(X_train,X_pred)
Y_test= tfidf_vectorizer.transform(data_test)
Y_pred=tfidf_vectorizer.transform(labels_test)
#print(tfidf_matrix_new)
classifier.predict(Y_test)
print(confusion_matrix(Y_test, Y_pred))  
print(classification_report(Y_test, Y_pred))    


