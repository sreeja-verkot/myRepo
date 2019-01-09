# load the iris dataset 
from sklearn.datasets import load_iris 
from sklearn.preprocessing import StandardScaler
iris = load_iris() 

# store the feature matrix (X) and response vector (y) 
X = iris.data 
y = iris.target 


# splitting X and y into training and testing sets 
from sklearn.cross_validation import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) 

scaler = StandardScaler()  

# every sklearn's transform's fit() just calculates the parameters (e.g. μ and σ in case of StandardScaler) and saves them as an internal objects state
scaler.fit(X_train)  

#X_train = scaler.transform(X_train)  
#X_test = scaler.transform(X_test)
# training the model on training set 
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB() 
gnb.fit(X_train, y_train) 

# making predictions on the testing set 
y_pred = gnb.predict(X_test) 

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
clf.fit(X_train, y_train)
y_pred2 = clf.predict(X_test) 
# comparing actual response values (y_test) with predicted response values (y_pred) 
from sklearn import metrics 
print("\nGaussian Naive Bayes model accuracy(in %):\n", metrics.accuracy_score(y_test, y_pred)*100)
print("\n Multinominal Naive Bayes model accuracy(in %):\n", metrics.accuracy_score(y_test, y_pred2)*100)

