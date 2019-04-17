import string
import preprocessor as p
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.stem.snowball import EnglishStemmer
p.set_options(p.OPT.URL, p.OPT.HASHTAG, p.OPT.MENTION, p.OPT.SMILEYL)
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
        text = p.clean(' '.join(text))
        clear_texts.append(text)
    return clear_texts
with open('/home/mtechcl/Desktop/sreeja/data/es_trial.text', 'r') as f:
    texts = [l for l in f]
    texts_clear = preproc_eng(texts)
    print(texts[5])
    print(texts_clear[5])
