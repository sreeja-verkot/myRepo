#https://machinelearningmastery.com/linear-regression-for-machine-learning/
#https://towardsdatascience.com/introduction-to-machine-learning-algorithms-linear-regression-14c4e325882a

#using scikit learn library to build our linear regression model
print("\nusing scikit learn library\n")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

df_train = pd.read_csv('train_linreg.csv')
df_test = pd.read_csv('test_linreg.csv')

#imputing Nan 
#df_train.fillna(df_train.mean(), inplace=True)
#df_test.fillna(df_train.mean(), inplace=True)

#dropping missing value rows
df_train.dropna(inplace=True)

x_train = df_train['x']
y_train = df_train['y']
x_test = df_test['x']
y_test = df_test['y']


#x_train = np.array(x_train)
#y_train = np.array(y_train)
#x_test = np.array(x_test)
#y_test = np.array(y_test)


x_train = x_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)


from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score

clf = LinearRegression(normalize=True)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print(r2_score(y_test,y_pred))
#print(clf.predict(np.array([24])))

plot.scatter(x_train, y_train, color = 'red')
plot.plot(x_train, clf.predict(x_train), color = 'blue')
plot.title('X vs Y (Training set)')
plot.xlabel('X')
plot.ylabel('Y')
plot.show()

plot.scatter(x_test, y_test, color = 'red')
plot.plot(x_train, clf.predict(x_train), color = 'blue')
plot.title('X vs Y (Test set)')
plot.xlabel('X')
plot.ylabel('Y')
plot.show()



