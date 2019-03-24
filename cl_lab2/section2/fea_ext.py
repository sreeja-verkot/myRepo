#https://scikit-learn.org/stable/modules/feature_extraction.html

print("\ncountvectorizer\n")
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
corpus = ['This is the first document.','This is the second second document.','And the third one.','Is this the first document?']
X = vectorizer.fit_transform(corpus)
#print(vectorizer.get_feature_names())
print(X.toarray())


print("\nTf–idf term weighting -In order to re-weight the count features into floating point values suitable for usage by a classifier it is very common to use the tf–idf transform. \n")
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=False)
counts = [[3, 0, 1],[2, 0, 0],[3, 0, 0],[4, 0, 0],[3, 2, 0],[3, 0, 2]]
tfidf = transformer.fit_transform(counts)
print(tfidf.toarray())

print("\nAs tf–idf is very often used for text features, there is also another class called TfidfVectorizer that combines all the options of CountVectorizer and TfidfTransformer in a single model:\n")
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X=vectorizer.fit_transform(corpus)
print(X.toarray())

print("\n feature extraction from dictionary- DictVectorizer\n")
measurements = [
    {'city': 'Dubai', 'temperature': 33.},
    {'city': 'London', 'temperature': 12.},
    {'city': 'San Francisco', 'temperature': 18.},
]

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
print(vec.fit_transform(measurements).toarray())
#print(vec.get_feature_names())

print("\n movie review classification using feature extraction\n")
import pandas as pd

#This data has 5 sentiment labels: 0 - negative 1 - somewhat negative 2 - neutral 3 - somewhat positive 4 - positive
data=pd.read_csv('train_fe.csv', sep='\t')
#print(data.head())

#Feature Generation using Bag of Words

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
#tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts= cv.fit_transform(data['Phrase'])

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(text_counts, data['Sentiment'], test_size=0.3, random_state=1)
from sklearn.naive_bayes import MultinomialNB
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("\nMultinomialNB Accuracy with BOW:\n",metrics.accuracy_score(y_test, predicted))

#Feature Generation using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer()
text_tf= tf.fit_transform(data['Phrase'])
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(text_tf, data['Sentiment'], test_size=0.3, random_state=123)
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("\nMultinomialNB Accuracy with tf-idf:\n",metrics.accuracy_score(y_test, predicted))
review=set()
i_p=input("Enter a movie review\n")
review.add(i_p)
print(list(review))
rev=tf.transform(list(review))
print(clf.predict(rev))
