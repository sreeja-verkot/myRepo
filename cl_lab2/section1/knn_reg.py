from sklearn.neighbors import KNeighborsRegressor
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import mean_squared_error,r2_score

df=pd.read_csv('reg.csv')
#print(df.head())

#read the columns except the last column 
X = df.iloc[:, :-1].values  

#read the last column that is class labels
y = df.iloc[:, -1].values 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 


#print(X_train)
#print(y_train)
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train, y_train) 
y_pred=knn.predict(X_test)
print("Mean squared error is :",mean_squared_error(y_test, y_pred))
print("r2 score is :",r2_score(y_pred,y_test))
print("For k=3 the weight for [5.5,38] is:",knn.predict([[5.5,38]]))
