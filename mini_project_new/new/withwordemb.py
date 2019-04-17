import re 
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score 
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
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
    X.append(text)
emotion_encoder = preprocessing.LabelEncoder()
y = emotion_encoder.fit_transform(y)
print(list(emotion_encoder.classes_))
sentences_train, sentences_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 

from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)

X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

print(sentences_train[2])
print(X_train[2])
from keras.preprocessing.sequence import pad_sequences
maxlen = 100
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
print(X_train[0, :])
from keras.models import Sequential
from keras import layers

embedding_dim = 50

model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size, 
                           output_dim=embedding_dim, 
                           input_length=maxlen))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
history = model.fit(np.array(X_train),np.array(y_train),epochs=50,batch_size=10)
loss, accuracy = model.evaluate(X_train, y_train)
print("Training loss: {:.4f}".format(loss))
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test)
print("Testing loss:  {:.4f}".format(loss))
