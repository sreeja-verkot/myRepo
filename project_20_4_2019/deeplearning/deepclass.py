import json
import numpy as np
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json

tokenizer = Tokenizer(num_words=3000)
#labels =["joy", 'fear', "anger", "sadness", "disgust", "shame", "guilt"]
labels=['anger', 'disgust', 'fear', 'guilt', 'joy', 'sadness', 'shame']
with open('dictionary.json', 'r') as dictionary_file:
    dictionary = json.load(dictionary_file)


def convert_text_to_index_array(text):
    words = kpt.text_to_word_sequence(text)
    wordIndices = []
    for word in words:
        if word in dictionary:
            wordIndices.append(dictionary[word])
        else:
            print("'%s' not in training corpus" %(word))
    return wordIndices

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('model.h5')

while 1:
    evalSentence = input("Enter a message:")

    if len(evalSentence) == 0:
        break

    testArr = convert_text_to_index_array(evalSentence)
    input1 = tokenizer.sequences_to_matrix([testArr], mode='binary')
    pred = model.predict(input1)
    print(pred)
    print(np.argmax(pred))
    print("%s ; %f%% confidence" % (labels[np.argmax(pred)], pred[0][np.argmax(pred)] * 100))
