#https://towardsdatascience.com/the-random-forest-algorithm-d457d499ffcd
#https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/

import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score

df=pd.read_csv('reg.csv')
#print(df.head())

#read the columns except the last column 
X = df.iloc[:, :-1].values  

#read the last column that is class labels
y = df.iloc[:, -1].values 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 

sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test)



regressor = RandomForestRegressor(n_estimators=20, random_state=0)  
regressor.fit(X_train, y_train)  
y_pred = regressor.predict(X_test)

print("Mean squared error is :",mean_squared_error(y_test, y_pred))
print("r2 score is :",r2_score(y_pred,y_test))
print("predicted weight for [5.5,38] is:",regressor.predict([[5.5,38]]))
