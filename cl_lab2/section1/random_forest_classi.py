from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from sklearn.datasets import load_iris 
from sklearn.preprocessing import StandardScaler
iris = load_iris() 


X = iris.data 
y = iris.target 

X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=0)  

sc=StandardScaler()
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test)  

classifier = RandomForestClassifier(n_estimators=20, random_state=0)  
classifier.fit(X_train,y_train)  
y_pred = classifier.predict(X_test)
print("\naccuracy score of random foresrt classifier:\n",metrics.accuracy_score(y_pred,y_test))  
print("class of [5.0,2.0,3.5,1.0] : ",classifier.predict([[5.0,2.0,3.5,1.0]]))