import pandas as pd  
from sklearn.cross_validation import train_test_split  
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('zoo.csv')
#We drop the animal names since this is not a good feature to split the data on
dataset=dataset.drop('animal_name',axis=1)
#Split the data into a training and a testing set
train_features = dataset.iloc[:80,:-1]
test_features = dataset.iloc[80:,:-1]
train_targets = dataset.iloc[:80,-1]
test_targets = dataset.iloc[80:,-1]
#Train the model
tree = DecisionTreeClassifier(criterion = 'entropy').fit(train_features,train_targets)
#Predict the classes of new, unseen data
prediction = tree.predict(test_features)
print("The prediction accuracy is: ",tree.score(test_features,test_targets)*100,"%")
