import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
df = pd.read_csv("/home/user/Desktop/mini_project/Train.csv")
X_all=df['TEXT']
Y=df['Label']
X=[]
import re 
for item in X_all:
    text_alphanum = re.sub('[^a-z0-9]', ' ', item)
    X.append(text_alphanum)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0) 
print(X_train[0])
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

#print(sentences_train[2])
#print(X_train[2])
from keras.preprocessing.sequence import pad_sequences
maxlen = 100
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
#print(X_train[0, :])
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
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()
history = model.fit(np.array(X_train),np.array(y_train),epochs=50,batch_size=10)
loss, accuracy = model.evaluate(X_train, y_train)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test)
print("Testing Accuracy:  {:.4f}".format(accuracy))
