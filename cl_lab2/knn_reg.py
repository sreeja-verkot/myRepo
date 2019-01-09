print("Example 1 :\n")
X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]
from sklearn.neighbors import KNeighborsRegressor
#from sklearn.cross_validation import train_test_split
#from sklearn.metrics import classification_report, confusion_matrix
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  
#neigh = KNeighborsRegressor(n_neighbors=1)
#neigh.fit(X_train, y_train) 
#y_pred=neigh.predict(X_test)
#print(X_test)
#print(confusion_matrix(y_test, y_pred))  
neigh = KNeighborsRegressor(n_neighbors=2)
neigh.fit(X,y) 
print(neigh.predict([[1.5]]))

print("\nExample 2:\n")
import pandas as pd  
from sklearn.cross_validation import train_test_split  
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix 
df=pd.read_csv('reg.csv')
df.head()
#read the columns except the last column 
X = df.iloc[:, :-1].values  

#read the last column that is class labels
y = df.iloc[:, -1].values 

#print(X)
#print(y)
for k in [3,5]:
   knn = KNeighborsRegressor(n_neighbors=k)
   knn.fit(X, y) 
   print("For k=",k,"the weight is:",knn.predict([[5.5,38]]))
