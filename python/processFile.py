import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
def process(content):
	
	stopwords_en = set(stopwords.words('english'))
	sents = nltk.sent_tokenize(content)
	wordList=[]
	for sent in sents:
		tokens = nltk.word_tokenize(sent)
		filtered_words = [w for w in tokens if not w in stopwords.words('english')]
		wordList.extend(filtered_words)
	print("hloo",wordList)
	return wordList 
