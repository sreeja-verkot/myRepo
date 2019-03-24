#https://machinelearningmastery.com/linear-regression-for-machine-learning/
#https://towardsdatascience.com/introduction-to-machine-learning-algorithms-linear-regression-14c4e325882a

#using scikit learn library to build our linear regression model
import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score
df=pd.read_csv('reg.csv')
#print(df.head())

#read the columns except the last column 
X = df.iloc[:, :-1].values  

#read the last column that is class labels
y = df.iloc[:, -1].values 
X_train, X_test, y_train, y_test = train_test_split(X,y,)




clf = LinearRegression(normalize=True)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("r2 score is :",r2_score(y_pred,y_test))
print("Mean squared error is :",mean_squared_error(y_test, y_pred))
print(clf.predict([[5.5,38]]))




