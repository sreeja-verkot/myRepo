#import the libraries

import re 
from collections import Counter
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score 
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer

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

#method to create features for the text messages
def create_feature(text, nrange=(1, 4)):
    text_features = [] 
    text = text.lower() 

    text_alphanum = re.sub('[^a-z0-9#]', ' ', text)
    #print(text_alphanum)
    for n in range(nrange[0], nrange[1]+1): 
        text_features += ngram(text_alphanum.split(), n)
    #print(text_features)
    return Counter(text_features)

#method to create ngrams(max 4 grams) because there can be words like 'not happy' 
def ngram(token, n): 
    #print("tokens:",token)
    #print("n:",n)
    output = []
    for i in range(n-1, len(token)): 
        ngram = ' '.join(token[i-n+1:i+1])
        output.append(ngram) 
    #print(output)
    return output


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
    X.append(create_feature(text))
#print(X[0],y[0])
"""emotion_encoder = LabelEncoder()
y = emotion_encoder.fit_transform(y)
print(list(emotion_encoder.classes_))"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 
#print(X_train,y_train)

vectorizer = DictVectorizer(sparse = True)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
svc=LinearSVC() 
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
print('\nAccuracy Score:\n')
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
