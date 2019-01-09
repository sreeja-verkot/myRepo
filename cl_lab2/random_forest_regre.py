#https://towardsdatascience.com/the-random-forest-algorithm-d457d499ffcd
#https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/

import pandas as pd  
import numpy as np  
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

print("\nRandom foresr regression  - predict the gas consumption based on petrol tax (in cents), per capita income (dollars), paved highways (in miles) and the proportion of population with the driving license.\n")

dataset = pd.read_csv('petrol_consumption.csv')  
dataset.head() 
X = dataset.iloc[:, 0:4].values  
y = dataset.iloc[:, 4].values  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 

sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test)



regressor = RandomForestRegressor(n_estimators=20, random_state=0)  
regressor.fit(X_train, y_train)  
y_pred = regressor.predict(X_test)

print(regressor.predict([[7.5,4399,431,.544]]))
print('\nMean Absolute Error:\n', metrics.mean_absolute_error(y_test, y_pred))  
print('\nMean Squared Error:\n', metrics.mean_squared_error(y_test, y_pred))  
print('\nRoot Mean Squared Error:\n', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  
#note :accuracy is a classification metric cant use with regression


