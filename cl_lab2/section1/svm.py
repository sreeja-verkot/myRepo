#https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/
#https://www.kaggle.com/nirajvermafcb/support-vector-machine-detail-analysis/notebook

#gender recognition using voice details
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris 
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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

svc=SVC() 

#Default hyperparameters
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
print('\nAccuracy Score with default hyperparameters:\n')
print(metrics.accuracy_score(y_test,y_pred))
scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy') #cv is cross validation
print("\ncross validation score:\n",scores)


