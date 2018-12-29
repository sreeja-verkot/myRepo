from nltk import FreqDist
from nltk.util import ngrams   
import nltk
from nltk.collocations import *
from nltk.tokenize import word_tokenize 
def compute_freq(content,n):
    tokens = word_tokenize(content)
    print(tokens)
    ngram = ngrams(tokens, n)
    for gram in ngram:
        print(gram)
    """bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(tokens)
    finder.apply_freq_filter(3)
    print(finder.nbest(bigram_measures.pmi, 5))"""
    
file_name=input("Enter the file name :\n")
file=open(file_name,'r+')
content=file.read()
print(content)
n=int(input("Enter n for ngram:\n"))
compute_freq(content,n)

