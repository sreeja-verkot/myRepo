import pandas as pd  
from sklearn.cross_validation import train_test_split  
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report, confusion_matrix 

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Assign colum names to the dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
dataset = pd.read_csv(url, names=names) 

#print(dataset.head())  

#read the columns except the last column 
X = dataset.iloc[:, :-1].values  

#read the last column that is class labels
y = dataset.iloc[:, -1].values 

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
print(classifier.predict([5.0,2.0,3.5,1.0]))
 
#print(confusion_matrix(y_test, y_pred))  
#print(classification_report(y_test, y_pred))    

