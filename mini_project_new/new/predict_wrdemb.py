#import the libraries

import re 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score 
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from gensim.models.word2vec import Word2Vec
import nltk
#method to read the training file and to seperate it into text and labels
def read_file(file_name): 
    data = []
    with open(file_name, 'r') as f: 
        for line in f: 
            #print(line)
            line = line.strip() 
            #print(line)
            label = ' '.join(line[1:line.find("]")].strip().split())
            text = line[line.find("]")+1:].strip()
            #print(label,text)
            data.append([label, text])
    return data 


#read the training data
file_name = "psychExp.txt"
psychExp_txt = read_file(file_name)
#print(psychExp_txt[0:3])
print("The number of instances: {}".format(len(psychExp_txt)))
#label generation
emotions = ["joy", 'fear', "anger", "sadness", "disgust", "shame", "guilt"]
for i in range(len(psychExp_txt)):
    item_new=psychExp_txt[i][0].replace('.','')
    item_new=item_new.replace(" ",'')
    #print(item_new)
    index=item_new.find("1")
    #print(index)
    label=emotions[index]
    #print(label)
    psychExp_txt[i][0]=label
#print(psychExp_txt)

X=[]
y=[]
for label, text in psychExp_txt:
    y.append(label)
    text_alnum=re.sub('[^a-z0-9]', ' ', text)
    X.append(text_alnum)
#print(X[0],y[0])

emotion_encoder = preprocessing.LabelEncoder()
y = emotion_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 
words=[]
for item in X_train:
    words.append(nltk.word_tokenize(item))
#print(words)
#print(len(words))
model=Word2Vec(words,min_count=1)
#print(model)
X = model[model.wv.vocab]
#print(words)
model.save("word2vec.model")
model = Word2Vec.load("word2vec.model")
model.train(X_train, total_examples=1, epochs=1)

#svm
svc=LinearSVC()
svc.fit(X,y_train)

print('\nAccuracy Score for svm:\n')
test=[]
for item in X_test:
    test.append(nltk.word_tokenize(item))
#print(words)
y_pred=svc.predict(Word2Vec(test))
print(metrics.accuracy_score(y_test,y_pred))
"""
#decision tree
tree=DecisionTreeClassifier(criterion = 'entropy') 
tree.fit(X_train,y_train)
y_pred=tree.predict(X_test)
print('\nAccuracy Score for decision tree:\n')
print(metrics.accuracy_score(y_test,y_pred))
#randomforest
rf=RandomForestClassifier(n_estimators=20, random_state=0)  
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)
print('\nAccuracy Score for random forest:\n')
print(metrics.accuracy_score(y_test,y_pred))

#naivebayes
nb=MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
nb.fit(X_train,y_train)
y_pred=nb.predict(X_test)
print('\nAccuracy Score for naivebayes:\n')
print(metrics.accuracy_score(y_test,y_pred))

#knn
knn=KNeighborsClassifier(n_neighbors=5)  
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
print('\nAccuracy Score for knn:\n')
print(metrics.accuracy_score(y_test,y_pred))

go_on=True
while(go_on==True):
    input_msg=input("\nenter a message\n")
    input_feature=create_feature(input_msg)
    print(svc.predict(vectorizer.transform([input_feature])))
    exit=input("\nDo you want to continue?y/n\n")
    if exit == "y":
        go_on=True
    else:
        go_on=False
"""
