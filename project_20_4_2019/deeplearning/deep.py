import json
import keras
import numpy as np
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from nltk.tokenize import  word_tokenize
from sklearn import preprocessing

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


file_name = "psychExp.txt"
psychExp_txt = read_file(file_name)
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

X=[]
y=[]
for label, text in psychExp_txt:
    y.append(label)
    X.append(text)
emotion_encoder = preprocessing.LabelEncoder()
y = emotion_encoder.fit_transform(y)
max_words = 3000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X)
dictionary = tokenizer.word_index
print(dictionary)
with open('dictionary.json', 'w') as dictionary_file:
    json.dump(dictionary, dictionary_file)


def convert_text_to_index_array(text):

    return [dictionary[word] for word in kpt.text_to_word_sequence(text)]

allWordIndices = []
for text in X:
    wordIndices = convert_text_to_index_array(text)
    allWordIndices.append(wordIndices)

print(allWordIndices[0])
allWordIndices = np.asarray(allWordIndices)
print(allWordIndices[0])
X = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')
y = keras.utils.to_categorical(y, 7)
print(X)

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 

model = Sequential()
model.add(Dense(512, input_shape=(max_words,), activation='relu'))
#model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.4))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy',  optimizer='adam',  metrics=['accuracy'])
model.fit(X_train, y_train,  batch_size=32,  epochs=250,  verbose=1,  validation_split=0.1,  shuffle=False,)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')



