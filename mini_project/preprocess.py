import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
import re
from nltk.stem.snowball import EnglishStemmer
translator = str.maketrans('', '', string.punctuation)
stemmer = EnglishStemmer()
def preproc_eng(texts):
    clear_texts = []
    for text in texts:
        # delete stop-words
        text = ' '.join([word for word in text.split() if word not in (stopwords.words('english'))])
        # delete punctuation
        text = word_tokenize(text.translate(translator))
        # stemming
        text = [stemmer.stem(w) for w in text]
        # preprocessing as tweet
        text =' '.join(text)
        text = re.sub(r'^https.* $', '', text)
        clear_texts.append(text)
    return clear_texts
with open('/home/user/Desktop/mini_project/train.text', 'r') as f:
    texts = [l for l in f]
    texts_clear = preproc_eng(texts)
    print(texts)
    print(texts_clear)
