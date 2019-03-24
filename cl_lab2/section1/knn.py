from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import accuracy_score 
# load the iris dataset 
from sklearn.datasets import load_iris 
from sklearn.preprocessing import StandardScaler
iris = load_iris() 

# store the feature matrix (X) and response vector (y) 
X = iris.data 
y = iris.target
#print(X)
#print(y)

#split the dataset to training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  


#scale the features

scaler = StandardScaler()  

# every sklearn's transform's fit() just calculates the parameters (e.g. μ and σ in case of StandardScaler) and saves them as an internal objects state
scaler.fit(X_train)  

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  
  
classifier = KNeighborsClassifier(n_neighbors=5)  
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("Accuracy is :",(accuracy_score(y_test, y_pred))) 
print("class of [5.0,2.0,3.5,1.0] : ",classifier.predict([[5.0,2.0,3.5,1.0]]))
 
#print(confusion_matrix(y_test, y_pred))  
#print(classification_report(y_test, y_pred))    

