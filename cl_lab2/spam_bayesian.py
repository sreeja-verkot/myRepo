from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np  
from math import log
import pandas as pd  #for loading data
import string
from collections import Counter

#load mails

mails=pd.read_csv('spam_nb.csv',encoding='latin-1')
mails.head()

#calculating the prior and conditional propbability of terms

N_spam = len(list(mails[mails['label'] == 1]['message']))
N_ham = len(list(mails[mails['label'] == 0]['message']))
N_files=mails.shape[0]
#print("no of spam :\n",N_spam)
#print("no of ham :\n",N_ham)
#print("total no of mails :\n",N_files)
stop_words = list(stopwords.words('english'))+list(string.punctuation)   
spam_msgs=word_tokenize(str(list(mails[mails['label'] == 1]['message'])))
terms_in_spam=[word.lower() for word in spam_msgs if word.lower() not in stop_words]
counts_spam = Counter(terms_in_spam)
#print ("no of words Spam vocab :\n", len(counts_spam))
ham_msgs=word_tokenize(str(list(mails[mails['label'] == 0]['message'])))
terms_in_ham=[word.lower() for word in ham_msgs if word.lower() not in stop_words]
counts_ham = Counter(terms_in_ham)
#print("no of words in ham mails:\n",len(terms_in_ham))
#print ("no of words ham vocab :\n", len(counts_ham))
all_msgs=word_tokenize(str(list(mails[mails['label'] == 1]['message'])+list(mails[mails['label'] == 0]['message'])))
terms_in_all = [word.lower() for word in all_msgs if word.lower() not in stop_words]
vocab = Counter(terms_in_all)
#print("Total number of terms in all files :\n", len(terms_in_all))
#print("Sze of vocab :\n", len(vocab))
P_spam = N_spam/float(N_files)
#print("Spam prior probability :",P_spam)
P_ham = N_ham/float(N_files)
#print("Ham prior probability :",P_ham)
cond_prob = {'spam': {}, 'ham': {}}
score_spam = log(P_spam)
score_ham = log(P_ham)
for term in vocab:
    term_spam_count = counts_spam[term] 
    term_ham_count = counts_ham[term]
    cond_prob['spam'][term] = (term_spam_count+1)/float(len(terms_in_spam)+len(vocab))
    cond_prob['ham'][term] = (term_ham_count+1)/float(len(terms_in_ham)+len(vocab))

class SpamClassifier:
    def __init__(self):
        self.prior_spam = None
        self.prior_ham = None
        self.likelihood = None
        
    def classify(self, message_terms):
        score_spam = self.prior_spam
        score_ham = self.prior_ham
        for term in message_terms:
            try:
                score_spam += log(self.likelihood['spam'][term])
            except KeyError as e:
                score_spam += log(1/float(len(terms_in_spam)+len(vocab)))
            try:
                score_ham += log(self.likelihood['ham'][term])
            except KeyError as e:
                score_ham += log(1/float(len(terms_in_ham)+len(vocab)))
        if score_spam > score_ham:
            return 1 #SPAM
        else:
            return 0 #HAM
clf = SpamClassifier()
clf.prior_spam = log(P_spam)
clf.prior_ham = log(P_ham)
clf.likelihood = cond_prob
msb_input=input("Enter a meassage:\n")
input_terms=word_tokenize(msb_input)
terms_in_input=[word.lower() for word in input_terms if word.lower() not in stop_words]
result=clf.classify(terms_in_input)
if result==1:
    print("Message is a spam")
else:
    print("Message is a ham")
