import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
def process(content):
	
	stopwords_en = set(stopwords.words('english'))
	sents = nltk.sent_tokenize(content)
	wordList=[]
	for sent in sents:
		sent = sent.lower()
		tokenizer = RegexpTokenizer(r'\w+')
		tokens = tokenizer.tokenize(sent)
		filtered_words = [w for w in tokens if not w in stopwords.words('english')]
		wordList.append(filtered_words)
	print filtered_words 
